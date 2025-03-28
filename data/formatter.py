with open('mg_scale.txt') as data:
	lines = data.readlines()

file2 = open(r'mg_scale_f.jl','w')
file2.write("A = [\n")

K = len(lines)

data = 1385
features = 6

Table = []

for line in lines:
	Table.append([str(0)]*features)
	entries = line.split()[1:]
	for entry in entries:
		entry_l, entry_r = entry.split(":")
		Table[-1][int(entry_l)-1] = entry_r
	row = ""
	for c in range(features):
		row += Table[-1][c]
		row += " "
	row += ";\n"
	file2.write(row)

file2.write("]")