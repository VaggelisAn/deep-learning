"""
Utilities to split MovieLens 100K `u.data` into train / validation / test sets
with held-out negatives.

Expected input format (tab-separated MovieLens 100K `u.data`):
user_id item_id rating timestamp

Outputs:
- ml-100k.train.rating: all interactions except the very last one per user
- ml-100k.test.rating: the last interaction per user
- ml-100k.validation.rating: one sampled interaction per user from the pre-test history
- ml-100k.test.negative: 100 sampled negative items per user, excluding training and test items
"""

from collections import defaultdict
import os
import random
import argparse
from typing import List, Tuple, Dict


def read_ratings(path: str) -> List[Tuple[int,int,float,int]]:
	ratings = []
	with open(path, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			if '\t' in line:
				parts = line.split('\t')
			else:
				parts = line.split()
			if len(parts) < 4:
				continue
			u = int(parts[0])
			i = int(parts[1])
			r = float(parts[2])
			t = int(parts[3])
			ratings.append((u, i, r, t))
	if not ratings:
		raise ValueError(f"No valid ratings found in {path}")
	min_user = min(r[0] for r in ratings)
	min_item = min(r[1] for r in ratings)
	if min_user < 1 or min_item < 1:
		raise ValueError(
			"Detected zero-based IDs or invalid format. This splitter expects MovieLens 100K "
			"Data/u.data format with 1-based user/item IDs."
		)
	return ratings


def write_list(path: str, records: List[Tuple[int,int,float,int]]):
	with open(path, 'w', encoding='utf-8') as f:
		for u,i,r,t in records:
			f.write(f"{u}\t{i}\t{int(r)}\t{t}\n")


def write_negative(path: str, negative_by_user: Dict[int, List[int]]):
	with open(path, 'w', encoding='utf-8') as f:
		for u in sorted(negative_by_user.keys()):
			negatives = negative_by_user[u]
			line = '\t'.join(str(i) for i in [u] + negatives)
			f.write(line + '\n')


def split_u_data(ratings: List[Tuple[int,int,float,int]], seed: int = 42):
	rng = random.Random(seed)
	by_user = defaultdict(list)
	item_ids = set()
	for u,i,r,t in ratings:
		by_user[u].append((u,i,r,t))
		item_ids.add(i)

	train = []
	validation = []
	test = []
	negative_by_user = {}
	all_items = set(range(1, max(item_ids) + 1))

	# Sort users to ensure consistent ordering between test.rating and test.negative
	for u in sorted(by_user.keys()):
		recs = by_user[u]
		recs_sorted = sorted(recs, key=lambda x: x[3])
		test_rec = recs_sorted[-1]
		test.append(test_rec)
		pre_test = recs_sorted[:-1]
		train.extend(pre_test)
		if pre_test:
			val_rec = rng.choice(pre_test)
			validation.append(val_rec)
		candidate_negatives = list(all_items - {r[1] for r in pre_test} - {test_rec[1]})
		if len(candidate_negatives) < 100:
			raise ValueError(f"Not enough negative candidates for user {u}")
		negative_by_user[u] = rng.sample(candidate_negatives, 100)

	return train, validation, test, negative_by_user


def split_file(input_path: str, out_dir: str, seed: int = 42):
	os.makedirs(out_dir, exist_ok=True)
	ratings = read_ratings(input_path)
	train, validation, test, negative_by_user = split_u_data(ratings, seed=seed)
	write_list(os.path.join(out_dir, 'ml-100k.train.rating'), train)
	# Sort validation and test by user ID for consistency with test.negative
	validation_sorted = sorted(validation, key=lambda x: x[0])
	test_sorted = sorted(test, key=lambda x: x[0])
	write_list(os.path.join(out_dir, 'ml-100k.validation.rating'), validation_sorted)
	write_list(os.path.join(out_dir, 'ml-100k.test.rating'), test_sorted)
	write_negative(os.path.join(out_dir, 'ml-100k.test.negative'), negative_by_user)
	return len(ratings), len(train), len(validation), len(test)


def main():
	parser = argparse.ArgumentParser(description='MovieLens 100K Leave-One-Out splitter')
	parser.add_argument('--input', '-i', default='Data/u.data', help='Input ratings file')
	parser.add_argument('--out', '-o', default='Data', help='Output directory')
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()
	total, ntrain, nvalidation, ntest = split_file(args.input, args.out, seed=args.seed)
	print(f"Total records: {total}")
	print(f"Train: {ntrain}, Validation: {nvalidation}, Test: {ntest}")


if __name__ == '__main__':
	main()

