import pynq
import numpy as np
import os
import shutil
from axi_stream_driver import NeuralNetworkOverlay

USER_PATH    = os.getcwd()

def main():
    #os.system("source /opt/xilinx/xrt/setup.sh")
    os.system("xbutil examine -d0000:00:10.1 -r thermal")

    # check the data
    X_test = np.load(USER_PATH + "/Data/X_test.npy")
    X_test = np.asarray(X_test, dtype = np.float32)

    # Load the xclbin file to the board
    ol = NeuralNetworkOverlay(xclbin_name="test4.xclbin")
    
    print(ol.krnl_rtl_1)
    print(ol.krnl_rtl_1.register_map)
    print(ol.krnl_rtl_1.signature)

    i_buff, o_buff = ol.allocate_mem(X_shape=X_test.shape, y_shape=(X_test.shape[0],10), dtype=np.float32, trg_in=ol.HBM0, trg_out=ol.HBM0)
    y, _, _ = ol.predict(X=X_test, y_shape=(X_test.shape[0],10), input_buffer=i_buff, output_buffer=o_buff, profile = True, debug=True)
    np.save( USER_PATH + "/Data/y_alveo_CNN.npy", y)
    
    '''
    i_buff, o_buff = ol.allocate_mem(X_shape=X_test.shape, y_shape=(X_test.shape[0],10), dtype=np.float32, trg_in=ol.HBM0, trg_out=ol.HBM0)
    N_it = 100
    rate_v  = []
    for i in range(N_it):
        N = int(((i+1)*X_test.shape[0]/N_it))
        i_buff[:N] = X_test[:N]
        y, _, rate = ol.predict(X=X_test[:N], y_shape=(N,10), input_buffer=i_buff[:N], output_buffer=o_buff[:N], profile=True, debug=True)
        rate_v.append(rate)
    '''

    y_alveo = y

    from sklearn.metrics import accuracy_score

    y_test = np.load(USER_PATH + "/Data/testY.npy")
    y_test.astype(np.float32)
 
    print("Accuracy FPGA: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_alveo, axis=1))))


# Call the Main function
if __name__ == '__main__':
	main()
