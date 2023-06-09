/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "xf_kalmanfilter_config.h"

namespace xf{
namespace cv{	

void kalmanfilter_SingleInstance(
						ap_uint<32>* in_A,
#if KF_C != 0
                        ap_uint<32>* in_B,
#endif
                        ap_uint<32>* in_Uq,
                        ap_uint<32>* in_Dq,
                        ap_uint<32>* in_H,
                        ap_uint<32>* in_X0,
                        ap_uint<32>* in_U0,
                        ap_uint<32>* in_D0,
                        ap_uint<32>* in_R,
#if KF_C != 0
                        ap_uint<32>* in_u,
#endif
                        ap_uint<32>* in_y,
                        unsigned char control_flag,
						bool loadX0UD0_en,
                        ap_uint<32>* out_X,
                        ap_uint<32>* out_U,
                        ap_uint<32>* out_D) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    ap_uint<8> flag = control_flag;

    xf::cv::Mat<TYPE, KF_N, KF_N, NPC1> A_mat;
#if KF_C != 0
    xf::cv::Mat<TYPE, KF_N, KF_C, NPC1> B_mat;
#endif
    xf::cv::Mat<TYPE, KF_N, KF_N, NPC1> Uq_mat;
    xf::cv::Mat<TYPE, KF_N, 1, NPC1> Dq_mat;
    xf::cv::Mat<TYPE, KF_M, KF_N, NPC1> H_mat;
    xf::cv::Mat<TYPE, KF_N, 1, NPC1> X0_mat;
    xf::cv::Mat<TYPE, KF_N, KF_N, NPC1> U0_mat;
    xf::cv::Mat<TYPE, KF_N, 1, NPC1> D0_mat;
    xf::cv::Mat<TYPE, KF_M, 1, NPC1> R_mat;
#if KF_C != 0
    xf::cv::Mat<TYPE, KF_C, 1, NPC1> u_mat;
	if(flag[1]==0)
		u_mat.size = 0;
#endif
    xf::cv::Mat<TYPE, KF_M, 1, NPC1> y_mat;
 /*   
	if(flag[2]==0)
	    y_mat.size = 0;
	if(loadX0UD0_en==1)
	{	
		A_mat.size = 0;
#if KF_C != 0		
		B_mat.size = 0;
#endif		
		Uq_mat.size = 0;
		Dq_mat.size = 0;
		H_mat.size = 0;
		R_mat.size = 0;
	}
*/
	if(flag[2]==0)
	    y_mat.rows = 0;
	if(loadX0UD0_en==1)
	{	
		A_mat.rows = 0;
#if KF_C != 0		
		B_mat.rows = 0;
#endif		
		Uq_mat.rows = 0;
		Dq_mat.rows = 0;
		H_mat.rows = 0;
		R_mat.rows = 0;
	}
	if(flag[0] == 0){
	    X0_mat.rows = 0;
	    U0_mat.rows = 0;
	    D0_mat.rows = 0;
	}
    xf::cv::Mat<TYPE, KF_N, 1, NPC1> Xout_mat;
    xf::cv::Mat<TYPE, KF_N, KF_N, NPC1> Uout_mat;
    xf::cv::Mat<TYPE, KF_N, 1, NPC1> Dout_mat;
	if(flag[3]==0 && flag[5] == 0)
	    Xout_mat.rows = 0;
	if(flag[4]==0 && flag[6] == 0){
	    Uout_mat.rows = 0;
	    Dout_mat.rows = 0;
	}

// clang-format off
    #pragma HLS STREAM variable=A_mat.data depth=2
#if KF_C != 0
    #pragma HLS STREAM variable=B_mat.data depth=2
#endif
    #pragma HLS STREAM variable=Uq_mat.data depth=2
    #pragma HLS STREAM variable=Dq_mat.data depth=2
    #pragma HLS STREAM variable=H_mat.data depth=2
    #pragma HLS STREAM variable=X0_mat.data depth=2
    #pragma HLS STREAM variable=U0_mat.data depth=2
    #pragma HLS STREAM variable=D0_mat.data depth=2
    #pragma HLS STREAM variable=R_mat.data depth=2
#if KF_C != 0
    #pragma HLS STREAM variable=u_mat.data depth=2
#endif
    #pragma HLS STREAM variable=y_mat.data depth=2
    #pragma HLS STREAM variable=Xout_mat.data depth=2
    #pragma HLS STREAM variable=Uout_mat.data depth=2
    #pragma HLS STREAM variable=Dout_mat.data depth=2
// clang-format on

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::accel_utils obj_inA, obj_inUq, obj_inU0, obj_inH, obj_inB;
    xf::cv::accel_utils obj_inDq, obj_inX0, obj_inD0, obj_inR, obj_iny;
    xf::cv::accel_utils obj_inu, obj_outU, obj_outD, obj_outX;

    // Retrieve xf::cv::Mat objects from img_in data:
      obj_inA.Array2xfMat<32, TYPE, KF_N, KF_N, NPC1>(in_A, A_mat);
      obj_inUq.Array2xfMat<32, TYPE, KF_N, KF_N, NPC1>(in_Uq, Uq_mat);
      obj_inH.Array2xfMat<32, TYPE, KF_M, KF_N, NPC1>(in_H, H_mat);
#if KF_C != 0
      obj_inB.Array2xfMat<32, TYPE, KF_N, KF_C, NPC1>(in_B, B_mat);
#endif
      obj_inDq.Array2xfMat<32, TYPE, KF_N, 1, NPC1>(in_Dq, Dq_mat);
      obj_inR.Array2xfMat<32, TYPE, KF_M, 1, NPC1>(in_R, R_mat);

#if KF_C != 0    
     obj_inu.Array2xfMat<32, TYPE, KF_C, 1, NPC1>(in_u, u_mat);
#endif
 
    obj_iny.Array2xfMat<32, TYPE, KF_M, 1, NPC1>(in_y, y_mat);
    obj_inU0.Array2xfMat<32, TYPE, KF_N, KF_N, NPC1>(in_U0, U0_mat);
    obj_inX0.Array2xfMat<32, TYPE, KF_N, 1, NPC1>(in_X0, X0_mat);
    obj_inD0.Array2xfMat<32, TYPE, KF_N, 1, NPC1>(in_D0, D0_mat);

    // Run xfOpenCV kernel:
#if KF_C != 0
    xf::cv::KalmanFilter<KF_N, KF_M, KF_C, KF_MTU, KF_MMU, XF_USE_URAM, 0, TYPE, NPC1, XF_USE_LUTRAM>(
        A_mat, B_mat, Uq_mat, Dq_mat, H_mat, X0_mat, U0_mat, D0_mat, R_mat, u_mat, y_mat, Xout_mat, Uout_mat, Dout_mat,
        flag, loadX0UD0_en);
#else

    xf::cv::KalmanFilter<KF_N, KF_M, KF_C, KF_MTU, KF_MMU, XF_USE_URAM, 0, TYPE, NPC1, XF_USE_LUTRAM>(
        A_mat, Uq_mat, Dq_mat, H_mat, X0_mat, U0_mat, D0_mat, R_mat, y_mat, Xout_mat, Uout_mat, Dout_mat,
        flag, loadX0UD0_en);
#endif
    obj_outU.xfMat2Array<32, TYPE, KF_N, KF_N, NPC1>(Uout_mat, out_U);
    obj_outD.xfMat2Array<32, TYPE, KF_N, 1, NPC1>(Dout_mat, out_D);
    obj_outX.xfMat2Array<32, TYPE, KF_N, 1, NPC1>(Xout_mat, out_X);

    return;
} // End of kernel

}}

extern "C" {

void kalmanfilter_accel(ap_uint<32>* in_A,
//                        ap_uint<32>* in_B,
                        ap_uint<32>* in_Uq,
                        ap_uint<32>* in_Dq,
                        ap_uint<32>* in_H,
                        ap_uint<32>* in_X0,
                        ap_uint<32>* in_U0,
                        ap_uint<32>* in_D0,
                        ap_uint<32>* in_R,
//                        ap_uint<32>* in_u,
                        ap_uint<32>* in_y,
                        unsigned char control_flag,
						int loop_cnt,
						bool loadX0UD0_en,
                        ap_uint<32>* out_X,
                        ap_uint<32>* out_U,
                        ap_uint<32>* out_D) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=in_A      offset=slave  bundle=gmem0
//    #pragma HLS INTERFACE m_axi      port=in_B      offset=slave  bundle=gmem1
    #pragma HLS INTERFACE m_axi      port=in_Uq     offset=slave  bundle=gmem2
    #pragma HLS INTERFACE m_axi      port=in_Dq     offset=slave  bundle=gmem3
    #pragma HLS INTERFACE m_axi      port=in_H      offset=slave  bundle=gmem4
    #pragma HLS INTERFACE m_axi      port=in_X0     offset=slave  bundle=gmem5
    #pragma HLS INTERFACE m_axi      port=in_U0     offset=slave  bundle=gmem6
    #pragma HLS INTERFACE m_axi      port=in_D0     offset=slave  bundle=gmem7
    #pragma HLS INTERFACE m_axi      port=in_R      offset=slave  bundle=gmem8
//    #pragma HLS INTERFACE m_axi      port=in_u      offset=slave  bundle=gmem9
    #pragma HLS INTERFACE m_axi      port=in_y      offset=slave  bundle=gmem10
    #pragma HLS INTERFACE m_axi      port=out_X     offset=slave  bundle=gmem11
    #pragma HLS INTERFACE m_axi      port=out_U     offset=slave  bundle=gmem12
    #pragma HLS INTERFACE m_axi      port=out_D     offset=slave  bundle=gmem13
    #pragma HLS INTERFACE s_axilite  port=control_flag 
    #pragma HLS INTERFACE s_axilite  port=loop_cnt     
    #pragma HLS INTERFACE s_axilite  port=loadX0UD0_en 
    #pragma HLS INTERFACE s_axilite  port=return       
    // clang-format on

	int offset_X=0;
	int offset_U=0;
	int offset_ctrl=0;
	int offset_y=0;
	
    for(int i=0; i < loop_cnt; i++)
    {
	  //fprintf(stderr,"\n[KF_ACCEL:]------------------------------kf Id:%d ------------------",i);
       xf::cv::kalmanfilter_SingleInstance
		( 
						in_A,
#if KF_C != 0
                        in_B,
#endif
                        in_Uq,
                        in_Dq,
                        in_H,
                        in_X0 + offset_X,
                        in_U0 + offset_U,
                        in_D0 + offset_X,
                        in_R,
#if KF_C != 0
                        in_u + offset_ctrl,
#endif
                        in_y + offset_y,
                        control_flag,
						loadX0UD0_en,
                        out_X + offset_X,
                        out_U + offset_U,
                        out_D + offset_X);
						
		offset_X+=(KF_N);
		offset_U+=(KF_N*KF_N);
		offset_ctrl+=(KF_C);
		offset_y+=(KF_M);				
    }


    return;
} // End of kernel

} // End of extern C
