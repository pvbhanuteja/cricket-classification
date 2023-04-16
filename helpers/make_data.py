from collections import defaultdict
import random

filename = 'data.txt'  # Replace with the name of your text file

with open(filename, 'r') as file:
    lines = [line.strip() for line in file.readlines()]

print(lines)

def get_species(line):
    if "Copy of" in line:
        return line.split("Copy of ")[1].split(" ")[0]
    return None

species_dict = defaultdict(list)

for line in lines:
    species = get_species(line)
    if species:
        species_dict[species].append(line)

set1, set2 = [], []

for species, items in species_dict.items():
    random.shuffle(items)
    split_index = int(len(items) * 0.8)
    set1.extend(items[:split_index])
    set2.extend(items[split_index:])

print("Set 1 (80%):")
print("\nSet 2 (20%):")

def save_to_file(filename, data):
    with open(filename, 'w') as file:
        for item in data:
            file.write(f"{item}\n")

save_to_file('train.txt', set1)
save_to_file('test.txt', set2)