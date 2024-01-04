import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set_style('white')
sns.set_palette('muted')
sns.set_context(
    "notebook",
    font_scale=1.25,
    rc={"lines.linewidth": 2.5}
)


def plot_roc(ax, fpr, tpr, auroc):
    ax.plot(fpr, tpr, 'b-', label=f'AUROC={auroc:.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='No Skill - 0.500')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.tick_params(axis='x', which='both', bottom=True)
    ax.tick_params(axis='y', which='both', left=True)
    ax.legend()


def plot_pr(ax, rec, pr, aupr, no_skill):
    ax.step(rec, pr, '-', where='post', label=f'AUPR={aupr:.3f}')
    ax.plot([0, 1], [no_skill, no_skill], linestyle='--', color='gray', label=f'No Skill - {no_skill:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.tick_params(axis='x', which='both', bottom=True)
    ax.tick_params(axis='y', which='both', left=True)
    ax.legend()


def add_labels(x, y, ax):
    for i in range(len(x)):
        ax.text(i, y[i], f"{y[i]:.3f}", ha='center')

