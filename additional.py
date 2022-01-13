

import timeit
convlist = []
for i in range(len(params)):
    if len(params[i].data.shape) == 4:
        convlist.append(i)

def cal_ch(params,convlist):
    
    total_channel = 0
    sparse_channel = 0
    for i in convlist:
        start = timeit.default_timer()
        #print("="*20)
        #print("Layer index:",i)
        total_channel += params[i].data.shape[0]
        #print("Layer",i,"Output channel cumulative",total_channel)
        for k in range(params[i].data.shape[0]):
            if (params[i].data[k,0,:,:] != 0).sum().item() > 0:
                continue
            elif (params[i].data[k,:,:,:] != 0).sum().item() == 0:
                sparse_channel += 1
                #print("Layer",i,"sparsity at ch",k)
        stop = timeit.default_timer()
        print('Time is',(stop-start))
    #print("="*40)
    channel_sparsity = sparse_channel/total_channel
    #print("Ratio: %d/%d" % (sparse_channel, total_channel))
    
    return channel_sparsity