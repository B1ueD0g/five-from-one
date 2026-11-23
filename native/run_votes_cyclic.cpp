#include "llama.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr int slots = 5;
constexpr int top_k = 8;
constexpr double temperature = 1.0;
constexpr int context_size = 4096;
constexpr int shared_prefix_size = 6;
constexpr int cpu_threads = 5;
using Ids = std::vector<llama_seq_id>;
using Tokens = std::vector<llama_token>;
using Clock = std::chrono::steady_clock;

struct Candidate { llama_token token; float logit; };
struct Scores { std::vector<Candidate> top; std::uint64_t fingerprint; };
struct Draw {
    llama_token token;
    int owner;
    std::uint64_t word;
    bool fresh;
    std::uint64_t logits;
};
enum class DecodeMode { parallel, serial };
enum class StopMode { eog, final_line };

struct Lane { std::vector<Draw> events; std::string terminal; };
struct Work { int prefill = 0; int first_prefix = 0; int tokens = 0; int calls = 0;
              int suffix_calls = 0; int max_batch = 0; };

std::string hex(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t derive_seed(int batch, int slot) {
    // SplitMix64 permutation of a fixed, indexed experiment key.
    std::uint64_t seed = 0x3cf174e90a18d526ULL + (batch * slots + slot + 1) * 0x9e3779b97f4a7c15ULL;
    seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9ULL;
    seed = (seed ^ (seed >> 27)) * 0x94d049bb133111ebULL;
    return seed ^ (seed >> 31);
}

Scores get_scores(llama_context *ctx, const llama_vocab *vocab, int output_index) {
    const float *logits = llama_get_logits_ith(ctx, output_index);
    require(logits != nullptr, "decode did not produce logits");
    Scores scores{{}, 14695981039346656037ULL};
    scores.top.reserve(top_k + 1);
    const auto better = [](Candidate left, Candidate right) {
        if (left.logit == right.logit) return left.token < right.token;
        return left.logit > right.logit;
    };
    for (llama_token token = 0; token < llama_vocab_n_tokens(vocab); ++token) {
        const float logit = logits[token];
        require(std::isfinite(logit), "nonfinite logit");
        std::uint32_t bits;
        std::memcpy(&bits, &logit, sizeof(bits));
        scores.fingerprint = (scores.fingerprint ^ bits) * 1099511628211ULL;
        Candidate entry{token, logit};
        if (scores.top.size() < top_k || better(entry, scores.top.back())) {
            auto at = std::lower_bound(scores.top.begin(), scores.top.end(), entry, better);
            scores.top.insert(at, entry);
            if (scores.top.size() > top_k) scores.top.pop_back();
        }
    }
    return scores;
}

llama_token pick(const Scores &scores, std::uint64_t word) {
    std::array<double, top_k> weights{};
    double total = 0;
    for (std::size_t index = 0; index < scores.top.size(); ++index) {
        const double logit = scores.top[index].logit;
        const double max_logit = scores.top.front().logit;
        weights[index] = std::exp((logit - max_logit) / temperature);
        total += weights[index];
    }
    double threshold = static_cast<double>(word >> 11) / 9007199254740992.0 * total;
    for (std::size_t index = 0; index + 1 < scores.top.size(); ++index) {
        if (threshold < weights[index]) return scores.top[index].token;
        threshold -= weights[index];
    }
    return scores.top.back().token;
}

Draw sample(const Scores &scores, int owner, std::uint64_t word) {
    return {pick(scores, word), owner, word, true, scores.fingerprint};
}

Scores decode(llama_context *ctx, const llama_vocab *vocab, const Tokens &tokens,
              const Ids &members, int position, Work &work) {
    auto batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, members.size());
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = position + i;
        batch.n_seq_id[i] = static_cast<int32_t>(members.size());
        for (std::size_t j = 0; j < members.size(); ++j) batch.seq_id[i][j] = members[j];
        batch.logits[i] = (i + 1 == batch.n_tokens);
    }
    int status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    require(status == 0, "llama_decode returned " + std::to_string(status));
    work.tokens += static_cast<int>(tokens.size());
    ++work.calls;
    return get_scores(ctx, vocab, -1);
}

using Tape = std::array<std::vector<std::uint64_t>, slots>;
using Lanes = std::array<Lane, slots>;

struct Item {
    int batch = 0;
    int replicate = 0;
    bool cold = false;
    std::string id;
    std::string split;
    std::string prompt;
    std::array<int, slots> source_batches{};
};

std::string unhex(const std::string &text) {
    require(text.size() % 2 == 0, "invalid hex");
    std::string bytes;
    for (std::size_t i = 0; i < text.size(); i += 2)
        bytes += static_cast<char>(std::stoi(text.substr(i, 2), nullptr, 16));
    return bytes;
}

std::string rendered_text(const llama_vocab *vocab, const Lane &lane) {
    Tokens ids;
    for (const auto &event : lane.events) {
        if (!llama_vocab_is_eog(vocab, event.token)) ids.push_back(event.token);
    }
    if (ids.empty()) return {};
    std::string text(4096, '\0');
    int len = llama_detokenize(vocab, ids.data(), ids.size(), text.data(), text.size(), false, true);
    if (len < 0) {
        text.resize(-len);
        len = llama_detokenize(vocab, ids.data(), ids.size(), text.data(), text.size(), false, true);
    }
    require(len >= 0, "cannot detokenize lane");
    text.resize(len);
    return text;
}

std::string text_hex(const std::string &text) {
    constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(text.size() * 2);
    for (unsigned char byte : text) {
        out += digits[byte >> 4];
        out += digits[byte & 15];
    }
    return out;
}

bool final_answer_body(std::string_view line) {
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) line.remove_prefix(1);
    while (!line.empty() && (line.back() == ' ' || line.back() == '\t')) line.remove_suffix(1);
    constexpr std::string_view marker = "Final: ";
    if (line.substr(0, marker.size()) != marker) return false;
    line.remove_prefix(marker.size());
    if (!line.empty() && line.front() == '-') line.remove_prefix(1);
    const auto scrapes_digits = [](std::string_view &text) {
        const auto before = text.size();
        while (!text.empty() && text.front() >= '0' && text.front() <= '9') text.remove_prefix(1);
        return text.size() != before;
    };
    if (!scrapes_digits(line)) return false;
    if (!line.empty() && line.front() == '.') {
        line.remove_prefix(1);
        if (!scrapes_digits(line)) return false;
    }
    return line.empty();
}

bool has_complete_answer(const std::string &text) {
    std::string_view view(text);
    while (true) {
        const auto newline = view.find('\n');
        if (newline == std::string_view::npos) return false;
        if (final_answer_body(view.substr(0, newline))) return true;
        view.remove_prefix(newline + 1);
    }
}

bool ends_in_answer_line(const llama_vocab *vocab, const Lane &lane) {
    // The engine only looks for completed lines. EOG is handled separately.
    auto token = lane.events.back().token;
    char small_piece[192];
    const int len = llama_token_to_piece(vocab, token, small_piece, sizeof(small_piece), 0, true);
    if (len >= 0 && std::find(small_piece, small_piece + len, '\n') == small_piece + len)
        return false;
    return has_complete_answer(rendered_text(vocab, lane));
}

void update_terminal(const llama_vocab *vocab, Lane &lane, int cap, StopMode stop) {
    if (llama_vocab_is_eog(vocab, lane.events.back().token)) lane.terminal = "eog";
    else if (stop == StopMode::final_line && ends_in_answer_line(vocab, lane))
        lane.terminal = "answer_line";
    else if (static_cast<int>(lane.events.size()) == cap) lane.terminal = "cap";
}

void feed_serial(llama_context *ctx, const llama_vocab *vocab, const Tape &tapes,
                 int slot, Lane &lane, Scores next, int root_size, int cap,
                 StopMode stop, Work &work) {
    while (lane.terminal.empty()) {
        const auto position = lane.events.size();
        const auto drawn = sample(next, slot, tapes[slot][position]);
        lane.events.push_back(drawn);
        update_terminal(vocab, lane, cap, stop);
        if (lane.terminal.empty()) {
            next = decode(ctx, vocab, {drawn.token}, {slot}, root_size + position, work);
            ++work.suffix_calls;
            work.max_batch = std::max(work.max_batch, 1);
        }
    }
}

void feed_parallel(llama_context *ctx, const llama_vocab *vocab, const Tape &tapes,
                   Lanes &lanes, const Scores &initial, int root_size, int cap,
                   StopMode stop, Work &work) {
    std::array<Scores, slots> scores;
    scores.fill(initial);
    auto abi = llama_batch_init(slots, 0, 1);
    for (int position = static_cast<int>(lanes[0].events.size()); position < cap; ++position) {
        std::array<int, slots> participating{};
        abi.n_tokens = 0;
        for (int slot = 0; slot < slots; ++slot) {
            auto &lane = lanes[slot];
            if (!lane.terminal.empty()) continue;
            const auto drawn = sample(scores[slot], slot, tapes[slot][position]);
            lane.events.push_back(drawn);
            update_terminal(vocab, lane, cap, stop);
            if (!lane.terminal.empty()) continue;
            const auto index = abi.n_tokens++;
            participating[index] = slot;
            abi.token[index] = drawn.token;
            abi.pos[index] = root_size + position;
            abi.n_seq_id[index] = 1;
            abi.seq_id[index][0] = slot;
            abi.logits[index] = true;
        }
        if (!abi.n_tokens) break;
        const int status = llama_decode(ctx, abi);
        if (status != 0) {
            llama_batch_free(abi);
            throw std::runtime_error("parallel llama_decode returned " + std::to_string(status));
        }
        ++work.calls;
        ++work.suffix_calls;
        work.tokens += abi.n_tokens;
        work.max_batch = std::max(work.max_batch, abi.n_tokens);
        for (int index = 0; index < abi.n_tokens; ++index)
            scores[participating[index]] = get_scores(ctx, vocab, index);
    }
    llama_batch_free(abi);
}

void write_arm(std::ostream &out, const Item &item, const Tokens &root, char mode,
               const std::array<std::uint64_t, slots> &seeds, const Lanes &lanes, Work work,
               const llama_vocab *vocab, double elapsed_ms, const Scores &initial) {
    out << "{\"kind\":\"batch\",\"item_id\":\"" << item.id << "\",\"split\":\"" << item.split
        << "\",\"replicate\":" << item.replicate << ",\"batch\":" << item.batch
        << ",\"mode\":\"" << mode << "\",\"seeds\":[";
    for (int slot = 0; slot < slots; ++slot)
        out << (slot ? "," : "") << '"' << hex(seeds[slot]) << '"';
    out << "],\"source_batches\":[";
    for (int slot = 0; slot < slots; ++slot)
        out << (slot ? "," : "") << item.source_batches[slot];
    out << "],\"prompt_tokens\":[";
    for (std::size_t i = 0; i < root.size(); ++i) out << (i ? "," : "") << root[i];
    out << "],\"prefill_tokens\":" << work.prefill << ",\"shared_generated_decode_tokens\":"
        << work.first_prefix << ",\"decoded_tokens\":" << work.tokens << ",\"decode_calls\":" << work.calls
        << ",\"suffix_decode_calls\":" << work.suffix_calls << ",\"max_suffix_batch_size\":" << work.max_batch
        << ",\"elapsed_ms\":" << elapsed_ms;
    if (mode == 'B' || mode == 'C') {
        out << ",\"root_logits_fingerprint\":\"" << hex(initial.fingerprint) << "\",\"root_top\":[";
        for (std::size_t i = 0; i < initial.top.size(); ++i) {
            const auto &[token, logit] = initial.top[i];
            std::uint32_t bits;
            std::memcpy(&bits, &logit, sizeof(bits));
            out << (i ? "," : "") << '[' << token << ",\"" << hex(bits).substr(8) << "\"]";
        }
        out << ']';
    }
    if (mode == 'C') {
        Lane prefix;
        const auto first = std::min<std::size_t>(lanes[0].events.size(), shared_prefix_size);
        prefix.events.assign(lanes[0].events.begin(), lanes[0].events.begin() + first);
        out << ",\"shared_prefix_text_hex\":\"" << text_hex(rendered_text(vocab, prefix)) << "\"";
        const bool terminal = first == lanes[0].events.size() && lanes[0].terminal != "cap";
        out << ",\"terminal_within_shared_prefix\":" << (terminal ? "true" : "false");
    }
    out << ",\"lanes\":[";
    for (int slot = 0; slot < slots; ++slot) {
        const auto &lane = lanes[slot];
        out << (slot ? "," : "") << "{\"slot\":" << slot << ",\"terminal\":\"" << lane.terminal
            << "\",\"text_hex\":\"" << text_hex(rendered_text(vocab, lane)) << "\"";
        if (lane.terminal == "answer_line")
            out << ",\"answer_line_end_token_position\":" << lane.events.size() - 1;
        out << ",\"tokens\":[";
        for (std::size_t i = 0; i < lane.events.size(); ++i)
            out << (i ? "," : "") << lane.events[i].token;
        out << "],\"events\":[";
        for (std::size_t i = 0; i < lane.events.size(); ++i) {
            const auto &event = lane.events[i];
            out << (i ? "," : "") << "{\"position\":" << i << ",\"token\":" << event.token
                << ",\"draw_owner\":" << event.owner << ",\"draw_word\":\"" << hex(event.word)
                << "\",\"new_draw\":" << (event.fresh ? "true" : "false")
                << ",\"logits\":\"" << hex(event.logits) << "\"}";
        }
        out << "]}";
    }
    out << "]}" << std::endl;
}

void run_arm(std::ostream &out, llama_context *ctx, const llama_vocab *vocab, const Item &item,
             const Tokens &root, char mode, const Tape &tapes,
             const std::array<std::uint64_t, slots> &seeds, int cap,
             DecodeMode execution, StopMode stop) {
    const auto start = Clock::now();
    auto *memory = llama_get_memory(ctx);
    Lanes lanes;
    Work work;
    Scores initial{{}, 0};
    if (mode == 'A') {
        for (int slot = 0; slot < slots; ++slot) {
            llama_memory_clear(memory, false);
            Scores next = decode(ctx, vocab, root, {slot}, 0, work);
            work.prefill += static_cast<int>(root.size());
            feed_serial(ctx, vocab, tapes, slot, lanes[slot], next, root.size(), cap, stop, work);
        }
    } else {
        llama_memory_clear(memory, false);
        Scores next = decode(ctx, vocab, root, {0}, 0, work);
        initial = next;
        for (int slot = 1; slot < slots; ++slot)
            llama_memory_seq_cp(memory, 0, slot, 0, root.size());
        work.prefill += static_cast<int>(root.size());
        if (mode == 'C') {
            for (int position = 0; position < shared_prefix_size && position < cap; ++position) {
                Draw event = sample(next, 0, tapes[0][position]);
                lanes[0].events.push_back(event);
                event.fresh = false;
                for (int slot = 1; slot < slots; ++slot) lanes[slot].events.push_back(event);
                update_terminal(vocab, lanes[0], cap, stop);
                if (!lanes[0].terminal.empty()) {
                    for (auto &lane : lanes) lane.terminal = lanes[0].terminal;
                    break;
                }
                if (position + 1 < cap) {
                    next = decode(ctx, vocab, {event.token}, {0}, root.size() + position, work);
                    ++work.first_prefix;
                }
            }
            if (lanes[0].terminal.empty()) {
                for (int slot = 1; slot < slots; ++slot)
                    llama_memory_seq_cp(memory, 0, slot, root.size(), root.size() + work.first_prefix);
            }
        }
        if (execution == DecodeMode::parallel)
            feed_parallel(ctx, vocab, tapes, lanes, next, root.size(), cap, stop, work);
        else
            for (int slot = 0; slot < slots; ++slot)
                feed_serial(ctx, vocab, tapes, slot, lanes[slot], next, root.size(), cap, stop, work);
    }
    const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    write_arm(out, item, root, mode, seeds, lanes, work, vocab, ms, initial);
}

std::array<int, slots> parse_source_batches(const std::string &line) {
    std::array<int, slots> result{};
    std::istringstream words(line);
    std::string text;
    for (int &batch : result) {
        require(static_cast<bool>(std::getline(words, text, ',')) && !text.empty()
                && std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; }),
                "invalid or missing tape-source batch");
        batch = std::stoi(text);
        require(batch <= 100000, "tape-source batch out of range");
    }
    require(words.eof(), "too many tape-source batches");
    return result;
}

std::vector<Item> load_items(const char *path) {
    std::ifstream stream(path);
    require(stream.good(), "could not open registered input");
    std::vector<Item> result;
    std::string line;
    while (std::getline(stream, line)) {
        std::array<std::string, 7> fields;
        std::istringstream row(line);
        for (auto &cell : fields)
            require(static_cast<bool>(std::getline(row, cell, '\t')), "invalid TSV row");
        result.push_back({std::stoi(fields[0]), std::stoi(fields[3]), fields[4] == "1",
                          fields[1], fields[2], unhex(fields[5]), parse_source_batches(fields[6])});
    }
    return result;
}

void run(const char *model_file, const std::string &model_id, const char *input, const char *output,
         int start, int count, int cap, DecodeMode execution, StopMode stop) {
    const auto items = load_items(input);
    require(cap >= 8 && cap <= 512, "cap must be between 8 and 512");
    require(!std::filesystem::exists(output), "output path already exists");
    require(model_id == "qwen3-4b" || model_id == "qwen25-3b", "unknown model identifier");
    require(start >= 0 && count > 0 && start + count <= static_cast<int>(items.size()), "invalid row range");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    auto *model = llama_model_load_from_file(model_file, mp);
    require(model != nullptr, "could not load model");
    auto cp = llama_context_default_params();
    cp.n_ctx = context_size;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_seq_max = slots;
    cp.kv_unified = true;
    cp.n_threads = cpu_threads;
    cp.n_threads_batch = cpu_threads;
    auto *ctx = llama_init_from_model(model, cp);
    require(ctx != nullptr, "could not create context");
    const auto *vocab = llama_model_get_vocab(model);

    std::ofstream out(output);
    require(out.good(), "could not open output");
    out << "{\"kind\":\"metadata\",\"model_id\":\"" << model_id << "\",\"llama_cpp\":\"b12428/b4c0e00a4\""
        << ",\"slot_count\":" << slots << ",\"cap\":" << cap << ",\"temperature\":" << temperature
        << ",\"top_k\":" << top_k << ",\"prefix_tokens\":" << shared_prefix_size << ",\"n_ctx\":" << context_size
        << ",\"n_threads\":" << cpu_threads << ",\"start\":" << start << ",\"count\":" << count
        << ",\"decode_mode\":\"" << (execution == DecodeMode::parallel ? "parallel" : "serial")
        << "\",\"stop_mode\":\"" << (stop == StopMode::final_line ? "final-line" : "eog")
        << "\",\"seed_key\":\"3cf174e90a18d526\",\"rng\":\"mt19937_64\",\"seed_schedule\":\"per-slot-source-batch-v1\"}" << std::endl;
    for (int index = start; index < start + count; ++index) {
        Item item = items[index];
        Tokens root(512);
        const int root_size = llama_tokenize(vocab, item.prompt.data(), item.prompt.size(),
                                              root.data(), root.size(), false, true);
        require(root_size > 0 && root_size <= 512, "failed tokenization or oversized prompt");
        root.resize(root_size);
        Tape tapes;
        std::array<std::uint64_t, slots> seeds{};
        for (int slot = 0; slot < slots; ++slot) {
            seeds[slot] = derive_seed(item.source_batches[slot], slot);
            std::mt19937_64 random(seeds[slot]);
            for (int token = 0; token < cap; ++token) tapes[slot].push_back(random());
        }
        std::string order = (item.batch % 2 == 0 ? "BC" : "CB");
        if (item.cold) order = (item.batch % 3 == 0 ? "ABC" : (item.batch % 3 == 1 ? "BCA" : "CAB"));
        for (const char mode : order) {
            try {
                run_arm(out, ctx, vocab, item, root, mode, tapes, seeds, cap, execution, stop);
                std::cerr << "completed batch=" << item.batch << " item=" << item.id << " mode=" << mode << '\n';
            } catch (const std::exception &e) {
                out << "{\"kind\":\"runtime_failure\",\"batch\":" << item.batch
                    << ",\"mode\":\"" << mode << "\"}" << std::endl;
                std::cerr << "failed batch=" << item.batch << " mode=" << mode << " " << e.what() << '\n';
                throw;
            }
        }
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 10) {
        std::cerr << "usage: run_votes MODEL MODEL_ID PROMPTS_TSV OUTPUT_JSONL FIRST COUNT CAP parallel|serial final-line|eog\n";
        return 2;
    }
    try {
        const std::string execution = argv[8];
        const std::string stop = argv[9];
        require(execution == "parallel" || execution == "serial", "invalid decode mode");
        require(stop == "final-line" || stop == "eog", "invalid stop mode");
        run(argv[1], argv[2], argv[3], argv[4], std::stoi(argv[5]), std::stoi(argv[6]), std::stoi(argv[7]),
            execution == "parallel" ? DecodeMode::parallel : DecodeMode::serial,
            stop == "final-line" ? StopMode::final_line : StopMode::eog);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
