import sys


dataset = 'Paris'

if dataset == 'Oxford':
    f = open('/imatge/ajimenez/workspace/ITR/lists/list_oxford.txt','r')
    f_q = open('/imatge/ajimenez/workspace/ITR/lists/queries_list_oxford.txt', 'r')
    f_o = open('/imatge/ajimenez/workspace/ITR/lists/list_oxford_no_queries.txt', 'w')

elif dataset == 'Paris':
    f = open('/imatge/ajimenez/workspace/ITR/lists/list_paris.txt','r')
    f_q = open('/imatge/ajimenez/workspace/ITR/lists/queries_list_paris.txt', 'r')
    f_o = open('/imatge/ajimenez/workspace/ITR/lists/list_paris_no_queries.txt', 'w')


name_list = list()

for query_name in f_q:
    name_list.append(query_name)


print name_list

count = 0
coincidences = 0
for line in f:
    flag = False
    if dataset == 'Oxford':
        line = line.replace('/imatge/ajimenez/work/datasets_retrieval/Oxford/1_images/', '')
    elif dataset == 'Paris':
        line = line.replace('/imatge/ajimenez/work/datasets_retrieval/Paris/imatges_paris/', '')
    line = line.replace('.jpg', '')
    sys.stdout.flush()
    for name in name_list:
        if line == name:
            print 'COINCIDENCE'
            coincidences +=1
            flag = True

    if not flag:
        count += 1
        line = line.replace('\n', '')
        if dataset == 'Oxford':
            f_o.write('/imatge/ajimenez/work/datasets_retrieval/Oxford/1_images/' + line + '.jpg' + '\n')
        elif dataset == 'Paris':
            f_o.write('/imatge/ajimenez/work/datasets_retrieval/Paris/imatges_paris/' + line + '.jpg' + '\n')

print count
print coincidences

f.close()
f_o.close()
f_q.close()
