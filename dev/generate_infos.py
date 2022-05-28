import os
import argparse


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ds', type=str, required=True)

	args = parser.parse_args()
	return args


def generate_infos(ds):
	assert ds in ['office_home', 'office31', 'visda17']
	folder = f'./data/{ds}/'
	out_dir = f'./data/{ds}_infos'
	os.makedirs(out_dir, exist_ok=True)

	domains = os.listdir(folder)
	domains.sort()

	for d in range(len(domains)):
		dom = domains[d]
		if os.path.isdir(os.path.join(folder, dom)):
			# replace ' ' with '_'
			dom_new = dom.replace(' ', '_')
			print(dom, dom_new)
			os.rename(os.path.join(folder, dom), os.path.join(folder, dom_new))
			dom_dir = os.path.join(folder, dom_new)

			paths = [os.path.join(dom_dir, f) for f in os.listdir(dom_dir)]
			clssses = [p for p in paths if os.path.isdir(p)]
			clssses.sort()
			out_path = os.path.join(out_dir, dom_new[0] + '_list.txt')
			f = open(out_path, 'w')
			for c in range(len(clssses)):
				cls = clssses[c]
				cls_new = cls.replace(' ', '_')
				print(cls, cls_new)
				os.rename(cls, cls_new)
				files = os.listdir(cls_new)
				files.sort()
				for file in files:
					file_new = file.replace(' ', '_')
					os.rename(os.path.join(cls_new, file), os.path.join(cls_new, file_new))
					print(file, file_new)
					print('{:} {:}'.format(os.path.join(cls_new, file_new), c))
					f.write('{:} {:}\n'.format(os.path.join(cls_new, file_new), c))
			f.close()


if __name__ == '__main__':
	args = get_args()
	generate_infos(args.ds)
