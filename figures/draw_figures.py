#!/usr/bin/env python3
"""Render conceptual diagrams and plot previously archived counts for Five from One?."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent

INK = '#182b39'
SUB = '#364b5b'
FRAME = '#ccd6de'
PALE = '#f0f5f8'
BLUE = '#285f83'
AQUA = '#e2eff8'
TEAL = '#26746f'
AMBER = '#a35520'
GOLD = '#f8e8d4'
GRAY = '#9babb8'
WHITE = '#ffffff'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'mathtext.fontset': 'dejavusans', 'text.color': INK,
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'axes.labelcolor': SUB, 'axes.edgecolor': FRAME,
    'xtick.color': SUB, 'ytick.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.titleweight': 'bold',
    'axes.titlesize': 9.3, 'hatch.linewidth': 0.45,
})


def emit(fig, key):
    fig.savefig(HERE / f'{key}.pdf', metadata={
        'Title': key.replace('_', ' ').title(), 'Author': None,
        'CreationDate': None, 'ModDate': None,
    })
    plt.close(fig)


def canvas(height):
    fig = plt.figure(figsize=(6.15, height))
    ax = fig.add_axes((0.01, 0.02, .98, .96), frame_on=False)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    return fig, ax


def box(ax, x, y, w, h, label, fc, ec, size=8.1):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                 boxstyle='round,pad=0.007,rounding_size=0.014',
                 linewidth=0.72, facecolor=fc, edgecolor=ec))
    ax.text(x, y, label, ha='center', va='center', fontsize=size, color=INK)


def arrow(ax, a, b, color=SUB, style='-', arc=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=8.5,
                 color=color, linestyle=style, linewidth=1.0,
                 connectionstyle=f'arc3,rad={arc}', zorder=1, shrinkA=0, shrinkB=0))


def draw_mechanisms():
    fig, ax = canvas(2.34)
    columns = ((.007, '(a) Shared prefill'), (.349, '(b) Shared sampled prefix'),
               (.691, '(c) Shared numerical state'))
    for left, title in columns:
        ax.text(left, .96, title, ha='left', va='top', weight='bold', fontsize=9)
    for sep in (.325, .667):
        ax.plot([sep, sep], [.07, .92], color=FRAME, lw=.65, zorder=0)

    for i, (left, _) in enumerate(columns):
        xs = (left+.073, left+.226)
        for j, x in enumerate(xs):
            box(ax, x, .769, .111, .102, rf'Tape $U_{j}$', WHITE, FRAME)
            box(ax, x, .14, .121, .104, rf'Slot $Y_{j}$', PALE, BLUE)
        mid = left+.15
        if i == 0:
            box(ax, mid, .44, .197, .106, 'Prompt state', AQUA, BLUE)
            for j, x in enumerate(xs):
                sgn = -1 if j == 0 else 1
                shift = .036 * sgn
                arrow(ax, (x+shift, .706), (x+shift, .206), INK)
                arrow(ax, (mid+.035*sgn, .374), (x, .205), BLUE)
        if i == 1:
            box(ax, xs[0], .44, .128, .106, 'Root prefix', GOLD, AMBER, 7.7)
            arrow(ax, (xs[0], .706), (xs[0], .504), AMBER)
            arrow(ax, (xs[0], .374), (xs[0], .206), INK)
            arrow(ax, (xs[1]+.028, .706), (xs[1]+.028, .206), INK)
            arrow(ax, (xs[0]+.071, .413), (xs[1]-.018, .207), AMBER)
            ax.text(mid+.011, .48, 'copy', ha='left', va='bottom', fontsize=7.6, color=AMBER)
        if i == 2:
            box(ax, mid, .44, .183, .106, 'Batch state', AQUA, BLUE)
            for j,x in enumerate(xs):
                sgn = -1 if j == 0 else 1
                shift = .037 * sgn
                arrow(ax,(x+shift,.706),(x+shift,.206),INK)
                arrow(ax,(x,.706),(mid+.033*sgn,.509),BLUE,'--')
                arrow(ax,(mid+.033*sgn,.374),(x,.206),BLUE,'--')
    emit(fig, 'mechanisms')


def array(ax, left, width, title, kind):
    cell = .043
    y0 = .64
    ygap = .104
    ax.text(left, .954, title, va='top', ha='left', fontsize=9.1, weight='bold')
    for j in range(width):
        ax.text(left+.106+cell*j, .777, str(j), va='center', ha='center', fontsize=8,
                color=SUB)
    for row in range(width):
        y = y0-ygap*row
        ax.text(left+.051, y, rf'${kind}^{row}$', va='center', ha='center', fontsize=9)
        for j in range(width):
            r = row if kind == 'A' else (row-j) % width
            x = left+.106+cell*j
            accent = r == 0
            ax.add_patch(Rectangle((x-cell*.48,y-.041),cell*.96,.082,
                         facecolor=AQUA if accent else WHITE,
                         edgecolor=BLUE if accent else FRAME,
                         linewidth=.88 if accent else .52))
            ax.text(x,y, str(r), va='center', ha='center',
                    color=BLUE if accent else SUB, fontsize=8.5,
                    weight='bold' if accent else 'normal')


def draw_design():
    fig, ax = canvas(2.35)
    width = 5
    array(ax, .007, width, '(a) Original calls', 'A')
    array(ax, .336, width, '(b) Remixed calls', 'R')
    ax.text(.686, .954, '(c) One slot per replay', va='top', ha='left',
            fontsize=9.1, weight='bold')
    others = (.653,)
    for sep in others:
        ax.plot([sep,sep],[.149,.912],color=FRAME,lw=.7,zorder=0)
    for sep in (.302,):
        arrow(ax,(sep,.465),(sep+.028,.465),BLUE)
    ax.text(.696,.756, r'Anchor $r=0$', fontsize=8.7, ha='left', va='center')
    for j in range(width):
        y=.636-.096*j
        ax.text(.705,y,rf'$Z^0_{j}$',fontsize=8.8,ha='left',va='center')
        ax.text(.777,y,'=',fontsize=8.8,ha='center',va='center',color=SUB)
        ax.text(.817,y,rf'$R^{j}_{j}$',fontsize=8.8,ha='left',va='center')
        ax.text(.909,y,rf'$U_{{0,{j}}}$',fontsize=8.8,ha='left',va='center',color=BLUE)
    ax.text(.012,.079,r'Cell $r$ in column $j$ represents tape $U_{r,j}$.',
            ha='left', va='center', color=SUB, fontsize=8.2)
    ax.text(.68,.079,'Readout   Replay   Retained tape',
            ha='left',va='center',fontsize=7.9,color=SUB)
    emit(fig, 'design')


def chart_axes(ax, xlim, tick, rows, ylim):
    ax.set_axisbelow(True)
    ax.set_xlim(0, xlim)
    ax.set_xticks(tick)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_yticks(range(len(rows)), rows)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()
    ax.tick_params(axis='both', length=0, pad=4, labelsize=8.4)
    ax.grid(axis='x', color=FRAME, alpha=.62, linewidth=.57)
    ax.axvline(0, color=SUB, lw=.55, clip_on=False)


def draw_results(counts):
    fig = plt.figure(figsize=(6.15,4.38))
    ax = fig.add_axes((.38,.595,.587,.308))
    fig.text(.014,.970,'(a) Terminal-sequence changes under tape controls',
             fontsize=9.6, weight='bold')
    handles=(Patch(facecolor=BLUE,edgecolor=BLUE,label='B  Slot-local draws'),
             Patch(facecolor=GOLD,edgecolor=AMBER,hatch='///',label='C  Shared sampled prefix'))
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.002,.944),
               frameon=False,ncol=2,fontsize=8.0,borderaxespad=.15,handlelength=1.4,
               handletextpad=.52,columnspacing=1.7)
    chart_axes(ax,.615,(0,.1,.2,.3,.4,.5),(
        'Serial, full-tape repeat',
        'Parallel, full-tape repeat',
        'Serial, outside-slot tapes changed',
        'Parallel, outside-slot tapes changed',
        'Serial vs. parallel, full tape',
    ),(-.50,4.50))
    paths = counts['path_counts']
    for i,pair in enumerate(paths):
        for j,(n,total) in enumerate(pair):
            y=i+(-.25 if j==0 else .25)
            pct=n/total
            face,edge,hatch=(BLUE,BLUE,None) if j==0 else (GOLD,AMBER,'///')
            if n:
                ax.barh(y,pct,height=.23,color=face,edgecolor=edge,hatch=hatch,lw=.75)
            else:
                ax.plot(0,y,marker='|',color=edge,ms=4.7,mew=1.,clip_on=False)
            ax.text(pct+.011,y,f'{n}/{total}',va='center',ha='left',fontsize=7.8)
    ax.set_xlabel('Matched slots with different terminal sequences',fontsize=8.7,labelpad=4)

    draw_first_tokens(fig, counts['first_token_witnesses'])
    fig.savefig(HERE / 'results.svg', metadata={'Date': None})
    emit(fig,'results')


def draw_first_tokens(fig, witnesses):
    if len(witnesses) != 3 or {w['item'] for w in witnesses} != {'GSM8K-test-0511'}:
        raise ValueError('First-token panel expects three recorded witnesses from one question')
    ax = fig.add_axes((.012,.062,.974,.397), xlim=(0,1), ylim=(0,1))
    ax.set_axis_off()
    ax.text(.0,.959,'(b) First token divergences at fixed draw',fontsize=9.6,weight='bold',va='center')
    ax.text(.994,.959,'GSM8K test 0511  ·  Parallel B',fontsize=8.4,color=SUB,ha='right',va='center')
    ax.axhline(.862, color=FRAME, lw=.75)
    columns = (.095, .253, .438, .690, .910)
    headers = ('Common prefix','Draw owner','Same draw word','Logits fingerprints','Chosen token IDs')
    for x, label in zip(columns,headers):
        ax.text(x,.763,label,fontsize=8.1,ha='center',color=SUB,va='center')
    for i,row in enumerate(witnesses):
        first = row['first']
        left,right = first['left'],first['right']
        if not (first['same_draw'] and first['logits_fingerprint_changed']
                and left['draw_word']==right['draw_word'] and left['draw_owner']==right['draw_owner']==row['slot']
                and left['logits'][:4]!=right['logits'][:4] and left['token']!=right['token']):
            raise ValueError('Recorded draw/selection comparison changed')
        y=.622-.153*i
        ax.add_patch(Rectangle((.004,y-.062),.992,.124,facecolor=PALE if i%2==0 else WHITE,linewidth=0))
        ax.text(columns[0],y,f"{first['common_tokens_before']} tokens",ha='center',va='center',fontsize=9)
        ax.text(columns[1],y,f"Slot {row['slot']}",ha='center',va='center',fontsize=9)
        ax.text(columns[2],y,left['draw_word'],ha='center',va='center',fontsize=8.6,fontfamily='monospace')
        ax.text(columns[3],y,f"{left['logits'][:4]}…  →  {right['logits'][:4]}…",ha='center',
                va='center',fontsize=8.8,fontfamily='monospace',color=SUB)
        for x,e,edge,fill in ((.862,left,BLUE,AQUA),(.958,right,AMBER,GOLD)):
            ax.add_patch(FancyBboxPatch((x-.028,y-.051),.056,.102,
                         boxstyle='round,pad=0.003,rounding_size=0.014',edgecolor=edge,facecolor=fill,lw=.65))
            ax.text(x,y,str(e['token']),ha='center',va='center',fontsize=9.1,weight='bold',color=INK)
        ax.text(.910,y,'→',ha='center',va='center',color=SUB,fontsize=9)
    ax.axhline(.221,color=FRAME,lw=.7)
    ax.text(.004,.125,'Original → replay with four companion tapes changed',fontsize=8.3,color=SUB,va='center')
    ax.text(.994,.125,'Fingerprint prefixes shown',fontsize=8.3,ha='right',color=SUB,va='center')


if __name__ == '__main__':
    counts = json.loads((HERE / 'reported_counts.json').read_text())
    draw_design()
    draw_results(counts)
