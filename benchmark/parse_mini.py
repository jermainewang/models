import sys
import collections

models = [('fc_data', 1, 0), 
          ('fc_data', 2, 0),
          ('fc_data', 2, 1),
          ('fc_data_slice', 2, 0),
          ('fc_data_slice', 2, 1),
          ('fc_model', 2, 0),
          ('fc_data_ff', 1, 0),
          ('fc_data_ff', 2, 0),
          ('fc_data_ff', 2, 1),
          ('fc_model_ff', 2, 0),
          ('fc_single_manual', 1, 0),
          ('fc_model_manual', 2, 0),
          ('fc_data_manual', 2, 0),
          ]

runtimes = {}
with open(sys.argv[1]) as f:
    token = 'Training '
    for line in f:
        index = line.find(token)
        if index == -1:
            continue
        line = line[index + len(token):]
        index = line.find('+')
        model = line[:index]
        line = line[index + 1:]
        index = line.find('+')
        nworkers = int(line[:index])
        line = line[index + 1:]
        index = line.find('+')
        gpu_aggr = int(line[:index])
        line = line[index + 1:]
        index = line.find('+')
        ws = int(line[:index])
        line = line[index + 1:]
        index = line.find(' ')
        bs = int(line[:index])
        line = line[line.find(', ') + 2:]
        runtime = float(line[:line.find(' ')])
        runtimes[(model, nworkers, gpu_aggr, ws, bs)] = runtime


with open("__out.csv", "w") as wf:
    for ws in (1024, 2048, 4096, 8192):
        wf.write('%d x %d weights (%dMB)\n' % (ws, ws, ws * ws / (2 ** 20)))
        for bs in [2 ** i for i in range(1, 14)]:
            wf.write('%d' % bs)
            for model in models:
                key = tuple(list(model) + [ws, bs])
                print(key)
                if key in runtimes:
                    wf.write(', %.3f' % runtimes[key])
                else:
                    wf.write(', N/A')
            wf.write('\n')
