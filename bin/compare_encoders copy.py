from joblib import Parallel, delayed
from sys import argv
from Bjontegaard import *
from subprocess import check_output, PIPE, STDOUT
from os import system
from os.path import isfile

def runEncoder(cmd, encoder1,encoder2, LOG):

	seqMap = {'NebutaFestival_10bit': 'Neb','FlowerVase': 'Flo','SteamLocomotiveTrain_10bit': 'StLo','SlideEditing': 'SliE', 'PeopleOnStreet': 'PoS' ,'ChinaSpeed':'Chin','RaceHorsesC':'RHC', 'BQSquare': 'BQS','BQMall':'BQM','BasketballDrill' : 'BDril','BasketballDrive':'BDrv','BQTerrace':'BQT',
			  'PartyScene':'PtS','ParkScene':'PkS','Cactus':'CAC','Kimono':'KIM', 'BasketballPass' : 'BPas', 'Johnny': 'Jon'}

	re_bitrate = '%d\s*a\s*(\d+.\d+)\s*' % (num_frames)
	re_psnr = '%d\s*a\s*\d+.\d+\s*(\d+.\d+)\s*(\d+.\d+)\s*(\d+.\d+)\s*(\d+.\d+)\s*' % (num_frames)
	re_time = 'Total\sTime:\s*(\d+.\d+)\s*'
	cmd_split = cmd.split()
	qp = int(cmd_split[cmd_split.index('-q')+1])
	if encoder1 in cmd:
		encoder = encoder1
	else:
		encoder = encoder2
	
	#print 'Running: ', cmd

	if cmd not in LOG.keys():
		print 'Running',outf_path
		output = check_output(cmd, shell=True,stderr=PIPE)

		bitrate = re.search(re_bitrate, output).group(1)
		time = re.search(re_time,output).group(1)
		[y,u,v,yuv] = re.search(re_psnr,output).groups()
	else:
		#print 'Running',outf_path
		[time,bitrate, y,u,v,yuv] = LOG[cmd]
	# ./hm-fast-fme-v1 -c ../cfg/encoder_randomaccess_main.cfg -c ~/hm-cfgs/BasketballDrill.cfg -q 37 -f 30



	#print encoder,qp,time,bitrate, y,u,v,yuv
	return [cmd,encoder,qp,time,bitrate, y,u,v,yuv]

encoder1 = argv[1]
test_encoders = argv[2].split(',')

sequences = argv[3].split(',')
num_frames = int(argv[4])
N_CORES =int(argv[5])

qps = [22,27,32,37]
gop_struct = 'encoder_randomaccess_main'



extra_opt1 = '--DecisionTree=0'
extra_opt2 = ['--DecisionTree=1',
			  '--DecisionTree=1 --Boosting=1'
			  ]

system('mkdir -p LOG')

LOG = {}

if isfile('LOG/logfile.txt'):
	logf = open('LOG/logfile.txt','r')

	for line in logf.readlines():
		lsplit = line.strip('\n').split()
		LOG[lsplit[0]] = lsplit[1:]

	logf.close()
	

for encoder2 in test_encoders:
	for seq in sequences:
		for extra_opt in extra_opt2:
			cmds = []
			for qp in qps:
				base_opt = '-c ../cfg/%s.cfg -c ~/hm-cfgs/cropped/%s.cfg -q %d -f %d' % (gop_struct, seq, qp, num_frames)
				cmd1 = './%s %s %s' % (encoder1, base_opt, extra_opt1)
				cmds.append(cmd1)

			for qp in qps:
				base_opt = '-c ../cfg/%s.cfg -c ~/hm-cfgs/cropped/%s.cfg -q %d -f %d' % (gop_struct, seq, qp, num_frames)
				cmd2 = './%s %s %s' % (encoder2, base_opt, extra_opt)
				cmds.append(cmd2)

			results = Parallel(n_jobs = N_CORES)( delayed (runEncoder)(cmd,encoder1,encoder2) for cmd in cmds )
			refBDResults = [[] for _ in qps]
			refTimes = [[] for _ in qps]
			testBDResults = [[] for _ in qps]
			avg_tr = 0.0
			for result in results:
				if result[0] not in LOG.keys():
					LOG[result[0]] = result[3:]
				qp = result[2]
				idx = qps.index(qp)
				time,bitrate, y,u,v,yuv = result[3:]

				if result[1] == encoder1:
					refBDResults[idx] = [float(bitrate),float(y),float(u),float(v),float(yuv)]
					refTimes[idx] = float(time)
				else:
					testBDResults[idx] = [float(bitrate),float(y),float(u),float(v),float(yuv)]
					tr = float(time)/float(refTimes[idx])
					avg_tr += tr		
			bdr = bdrate(refBDResults, testBDResults)		
			print '%s\t%d\t%s\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.2f\t%.2f' % (seq,num_frames,encoder2,extra_opt,bdr[0],bdr[1],bdr[2],bdr[3],  avg_tr/len(qps),1-(avg_tr/len(qps)))

logf = open('LOG/logfile.txt','w')
for k in LOG.keys():
	print >> logf, '\t'.join([k]+LOG[k])
