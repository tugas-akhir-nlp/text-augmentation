import re
import multiprocessing as mp
import timeit


# In[2]:
filename = '../corpus_path.txt'

with open('../corpus_path.txt', 'r') as file:
    path = file.readlines()[0].strip()
print(path)


# In[3]:


def number_masking(sentence):
    sentence = re.sub(r'[0-9]+', r'<NUM>', sentence)
    
    return sentence


start = timeit.default_timer()

with open(path) as file:
    corpus = file.readlines()

for i in range(len(corpus)):
    corpus[i] = number_masking(corpus[i])
# pool = mp.Pool(processes=4)
# results = [pool.apply(number_masking, args=(x,)) for x in corpus[:10]]
# output = [p.get() for p in results]
# print(output)

stop = timeit.default_timer()
execution_time = stop - start
print('execution time: {}'.format(execution_time))

with open(filename[:-4]+'_number_masked.txt', 'w') as file:
    file.writelines(corpus)