import os
import matplotlib.pyplot as plt

def _save_or_show(output_dir, filename):
    ''' Save the current figure to output_dir as PDF and PNG, or show interactively if output_dir is None. '''
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{filename}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'{filename}.png'), bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
