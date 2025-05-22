# 入力ファイルと出力ファイルのパスを設定
input_file = './results/samurai/samurai_base_plus/person-13.txt'  # 変換前のファイル名
output_file = './results/samurai/samurai_base_plus/person-13 copy.txt'  # 変換後のファイル名

# 入力ファイルを読み込む
with open(input_file, 'r') as f:
    lines = f.readlines()

# 出力用のデータを格納するリスト
output_lines = []

# 各行を変換
for i, line in enumerate(lines, start=1):
    # 行をカンマで分割してリストに変換
    parts = line.strip().split(',')

    # 1行目は1から始め、最後に指定された形式で追加
    output_line = f"{i},1,{','.join(parts)},1,-1,-1,-1"

    # 変換後の行をリストに追加
    output_lines.append(output_line)

# 出力ファイルに書き込む
with open(output_file, 'w') as f:
    f.write('\n'.join(output_lines))

print(f"変換が完了しました。結果は {output_file} に保存されています。")
