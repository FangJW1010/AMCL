import random
import numpy as np
def load_data(file_path):
    """
    从给定的文件路径加载数据。
    每对行：第一行是标签，第二行是序列。
    """
    data = []
    labels = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            if lines[i].strip():  # 确保标签行不为空
                label = np.array([int(x) for x in lines[i].strip()[1:]])
                sequence = lines[i + 1].strip() if i + 1 < len(lines) else ''
                if sequence:  # 确保序列行不为空
                    labels.append(label)
                    data.append(sequence)
    return data, labels


def augment_peptide_dataset_with_reversals_and_back_translation(data, labels, N):
    """
    对肽段数据集进行增强，包括反转、随机打乱和反向翻译增强。
    Args:
        data: 原始肽段序列列表。
        labels: 原始标签列表。
        N: 每个肽段生成的掩码和反转样本对数。
    Returns:
        augmented_peptides: 增强后的肽段列表。
        augmented_labels: 增强后的标签列表。
    """
    codon_table = {
        'A': ['GCU', 'GCC', 'GCA', 'GCG'],
        'C': ['UGU', 'UGC'],
        'D': ['GAU', 'GAC'],
        'E': ['GAA', 'GAG'],
        'F': ['UUU', 'UUC'],
        'G': ['GGU', 'GGC', 'GGA', 'GGG'],
        'H': ['CAU', 'CAC'],
        'I': ['AUU', 'AUC', 'AUA'],
        'K': ['AAA', 'AAG'],
        'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],
        'M': ['AUG'],
        'N': ['AAU', 'AAC'],
        'P': ['CCU', 'CCC', 'CCA', 'CCG'],
        'Q': ['CAA', 'CAG'],
        'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
        'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],
        'T': ['ACU', 'ACC', 'ACA', 'ACG'],
        'V': ['GUU', 'GUC', 'GUA', 'GUG'],
        'W': ['UGG'],
        'Y': ['UAU', 'UAC'],
    }
    reverse_codon_table = {v: k for k, values in codon_table.items() for v in values}
    # 定义氨基酸相似性字典
    # 这里根据侧链的化学性质进行分组
    similar_amino_acids = {
        'A': ['G', 'V', 'L'],  # 小型非极性
        'R': ['K', 'H'],  # 基础氨基酸
        'N': ['Q'],  # 酸性中性
        'D': ['E'],  # 酸性
        'C': ['S', 'T'],  # 硫醇/羟基
        'Q': ['N'],  # 酸性中性
        'E': ['D'],  # 酸性
        'G': ['A', 'V', 'L'],  # 小型非极性
        'H': ['R', 'K'],  # 基础氨基酸
        'I': ['L', 'V'],  # 分支链非极性
        'L': ['I', 'V'],  # 分支链非极性
        'K': ['R', 'H'],  # 基础氨基酸
        'M': ['I', 'L', 'V'],  # 大型非极性
        'F': ['Y', 'W'],  # 芳香族
        'P': ['G'],  # 环状
        'S': ['C', 'T'],  # 硫醇/羟基
        'T': ['S', 'C'],  # 硫醇/羟基
        'W': ['F', 'Y'],  # 芳香族
        'Y': ['F', 'W'],  # 芳香族
        'V': ['A', 'G', 'L'],  # 小型非极性
    }

    # 确保每个氨基酸至少有一个相似替换
    for aa in codon_table.keys():
        if aa not in similar_amino_acids or not similar_amino_acids[aa]:
            similar_amino_acids[aa] = [aa]

    augmented_peptides = []
    augmented_labels = []
    i = 0
    for peptide, peptide_label in zip(data, labels):
        # 添加原始序列和标签
        augmented_peptides.append(peptide)
        augmented_labels.append(peptide_label[:])
        positions_to_check = list(range(21))  # 检查前 21 个位置

        # 使用 any() 函数来判断是否需要增强
        should_augment = any(peptide_label[pos] == 1 for pos in positions_to_check if pos < len(peptide_label))

        if should_augment and np.sum(peptide_label) == 1:
            i += 1
            # 生成反转序列
            augmented_peptides.append(peptide[::-1])
            augmented_labels.append(peptide_label[:])

            for _ in range(N - 1):
                # 随机选择一个位置进行替换
                if len(peptide) > 0:
                    mask_position = random.randint(0, len(peptide) - 1)
                    original_aa = peptide[mask_position]
                    # 获取相似的氨基酸列表，排除原氨基酸
                    possible_replacements = [aa for aa in similar_amino_acids.get(original_aa, []) if aa != original_aa]
                    if possible_replacements:
                        new_aa = random.choice(possible_replacements)
                        masked_peptide = peptide[:mask_position] + new_aa + peptide[mask_position + 1:]
                        augmented_peptides.append(masked_peptide)
                        augmented_labels.append(peptide_label[:])  # 复制标签

            # 反向翻译成mRNA，进行随机替换，再翻译成多肽序列
            mrna_sequence = []
            for amino_acid in peptide:
                if amino_acid in codon_table:
                    codon = random.choice(codon_table[amino_acid])
                    mrna_sequence.append(codon)
            mrna_sequence = ''.join(mrna_sequence)

            # 确保mRNA序列长度大于等于3个碱基
            if len(mrna_sequence) >= 3:
                # 确保有其他非同义的密码子用于替换
                replacement_attempts = 0
                while True:
                    replacement_position = random.randint(0, len(mrna_sequence) // 3 - 1)
                    codon_start = replacement_position * 3
                    original_codon = mrna_sequence[codon_start:codon_start + 3]

                    if original_codon in reverse_codon_table:
                        # 原始密码子的氨基酸
                        original_amino_acid = reverse_codon_table[original_codon]

                        # 找到所有不同的氨基酸及其对应的密码子
                        different_codons = []
                        for amino_acid, codons in codon_table.items():
                            if amino_acid != original_amino_acid:  # 排除原始氨基酸
                                different_codons.extend(codons)

                        # 如果存在可用的不同义密码子
                        if different_codons:
                            new_codon = random.choice(different_codons)
                            mrna_sequence = mrna_sequence[:codon_start] + new_codon + mrna_sequence[codon_start + 3:]
                            break

                    # 防止无限循环，最多尝试 10 次
                    replacement_attempts += 1
                    if replacement_attempts >= 10:
                        break

            # 将mRNA序列翻译回多肽序列
            translated_peptide = ''
            for i in range(0, len(mrna_sequence), 3):
                codon = mrna_sequence[i:i + 3]
                if codon in reverse_codon_table:
                    translated_peptide += reverse_codon_table[codon]

            # 添加反向翻译替换后的序列和标签
            if translated_peptide:
                augmented_peptides.append(translated_peptide)
                augmented_labels.append(peptide_label[:])

    # 完成所有增强后，对整个增强数据集进行随机打乱
    augmented_data = list(zip(augmented_peptides, augmented_labels))
    random.shuffle(augmented_data)

    # 然后分别提取肽段和标签
    augmented_peptides, augmented_labels = zip(*augmented_data)

    # 返回打乱后的肽段和标签
    return list(augmented_peptides), list(augmented_labels)


def save_augmented_dataset(file_path, peptides, labels):
    """
    保存扩展的数据集到指定文件路径。
    Args:
        file_path: 保存文件的路径。
        peptides: 增强后的肽段序列列表。
        labels: 增强后的标签列表。
    """
    with open(file_path, 'w') as file:
        for label, peptide in zip(labels, peptides):
            label_str = ">" + "".join(map(str, label))  # 保持标签为整数独热编码格式
            file.write(label_str + "\n")
            file.write(peptide + "\n")

# 设置随机种子保证每次运行结果相同
random.seed(2024)
np.random.seed(2024)
# 主流程
input_file = 'TP_test/dataset/train.txt'  # 输入文件的路径
output_file = 'TP_test/dataset/augmented_train.txt'  # 保存扩展数据集的路径
# 加载原始数据
data, labels = load_data(input_file)
# 执行数据增强
N = 3
augmented_peptides, augmented_labels = augment_peptide_dataset_with_reversals_and_back_translation(data, labels, N)
# 保存扩展数据集
save_augmented_dataset(output_file, augmented_peptides, augmented_labels)
print(f"数据增强完成，扩展的数据集已保存至 '{output_file}'")

