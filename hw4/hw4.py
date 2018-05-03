import sys
import numpy as np
import pandas as pd
test_data = pd.read_csv(sys.argv[2])
IDs , idx1 , idx2 = np.array(test_data['ID']) , np.array(test_data['image1_index']), np.array(test_data['image2_index'])
fout = open(sys.argv[3] , 'w')
fout.write('ID,Ans\n')
for idx,i1,i2 in zip(IDs,idx1,idx2):
	fout.write("{},{}\n".format(idx,0))
fout.close()
