import matplotlib.pyplot as plt
import re
import os
import argparse

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-paper')


def parse_txt(file_path):
    gens = []
    accs = []
    if not os.path.exists(file_path):
        print(f" [Skip] File not found: {file_path}")
        return [], []

    num_pattern = re.compile(r"(\d+\.?\d*)")
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        matches = num_pattern.findall(line)
        if len(matches) >= 2:
            gens.append(int(matches[0]))
            accs.append(float(matches[1]))
        elif len(matches) == 1:
            gens.append(i)
            accs.append(float(matches[0]))

    return gens, accs


def plot_single_dataset(data_name):
    base_dir = f"/MobilenetV3/ENAS/{data_name}/"

    files = {
        "CDAP": {"path": "CDAP.txt", "color": "#d62728"},
        "MetaD2A": {"path": "MetaD2A.txt", "color": "#1f77b4"},
        "DiffusionNAG": {"path": "DiffusionNAG.txt", "color": "#008000"}
    }

    plt.figure(figsize=(7, 6), dpi=150)

    any_data = False
    for label, info in files.items():
        full_path = os.path.join(base_dir, info["path"])
        gens, accs = parse_txt(full_path)

        if not gens: continue
        any_data = True

        plt.plot(gens, accs, color=info["color"], linewidth=2.5, alpha=0.9, label=label)

        marker_gens = []
        marker_accs = []
        for i in range(len(accs)):
            if i == 0 or accs[i] != accs[i - 1]:
                marker_gens.append(gens[i])
                marker_accs.append(accs[i])
        plt.scatter(marker_gens, marker_accs, color=info["color"], s=40, edgecolors='white', zorder=5)

    if not any_data:
        print(f"Error: No valid data files found under directory {base_dir}.")
        return

    plt.xlabel("Generation", fontsize=24)
    plt.ylabel("Top-1 Accuracy (%)", fontsize=24)
    plt.title(data_name.upper(), fontsize=34)
    plt.xticks(range(0, 101, 20), fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=16, loc='lower right')

    output_path = os.path.join(base_dir, f"evolution_comparison_{data_name}.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"success! Image saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, nargs='+', default=['mnist'],
                        help='mnist cifar10 cifar100 aircraft pets svhn')
    args = parser.parse_args()

    for ds in args.data_name:

        plot_single_dataset(ds)
