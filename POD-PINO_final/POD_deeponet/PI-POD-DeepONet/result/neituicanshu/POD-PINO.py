# -*- coding: utf-8 -*-

"""
KM Physics Residual Calculation

修正版：
1. 防止NaN污染RMSE
2. 去除边界异常点
3. 保留R1/R2详细结果
4. 输出有效残差比例

"""

import os
import re
import numpy as np
import pandas as pd



# ==========================================================
# 路径
# ==========================================================

INPUT_ROOT_DIR = r"D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\SI_result_physics_informed_with_KM"


OUTPUT_ROOT_DIR = r"D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\result\waituicancha\result_5"


os.makedirs(
    OUTPUT_ROOT_DIR,
    exist_ok=True
)


CSV_NAME = "KM_coefficients_all_tau.csv"


SUMMARY_NAME = "physics_residual_summary.csv"



# ==========================================================
# 参数解析
# ==========================================================

def parse_folder_params(folder_name):

    nums = re.findall(
        r'\d+_\d+',
        folder_name
    )

    if len(nums)<3:
        return None


    return [
        float(x.replace("_","."))
        for x in nums[:3]
    ]



# ==========================================================
# 残差计算
# ==========================================================


def calculate_physics_residual(csv_file):


    df=pd.read_csv(csv_file)


    # 清理列名
    df.columns=df.columns.str.strip()



    # --------------------------
    # A
    # --------------------------

    A=df["A_plot"].astype(float).values



    D1_th=df[
        "D1_theoretical_opt"
    ].astype(float).values


    D2_th=df[
        "D2_theoretical_opt"
    ].astype(float).values



    # --------------------------
    # tau列
    # --------------------------

    tau_columns=[]


    for col in df.columns:


        m=re.search(
            r"D1_deeponet_opt_tau_(\d+\.\d+)s",
            col
        )


        if m:
            tau_columns.append(col)



    tau_columns=sorted(
        tau_columns,
        key=lambda x:
        float(
            re.findall(
                r'\d+\.\d+',
                x
            )[0]
        )
    )


    tau_list=np.array(
        [
            float(
                re.findall(
                    r'\d+\.\d+',
                    c
                )[0]
            )
            for c in tau_columns
        ]
    )



    n_tau=len(tau_list)

    n_A=len(A)



    if n_tau<3:
        raise ValueError(
            "tau数量不足"
        )



    # --------------------------
    # 构造D矩阵
    # --------------------------

    D1=np.zeros(
        (n_tau,n_A)
    )

    D2=np.zeros(
        (n_tau,n_A)
    )



    for i,col in enumerate(tau_columns):


        D1[i,:]=df[col].values



        d2_col=col.replace(
            "D1_fp",
            "D2_fp"
        )


        D2[i,:]=df[d2_col].values



    Tau=tau_list



    # ======================================================
    # U
    # ======================================================

    U1=Tau[:,None]*D1

    U2=Tau[:,None]*D2



    # ======================================================
    # 导数
    # ======================================================

    dU1_dA=np.gradient(
        U1,
        A,
        axis=1,
        edge_order=2
    )


    dU2_dA=np.gradient(
        U2,
        A,
        axis=1,
        edge_order=2
    )


    d2U1_dA2=np.gradient(
        dU1_dA,
        A,
        axis=1,
        edge_order=2
    )


    d2U2_dA2=np.gradient(
        dU2_dA,
        A,
        axis=1,
        edge_order=2
    )



    dU1_dtau=np.gradient(
        U1,
        Tau,
        axis=0,
        edge_order=2
    )


    dU2_dtau=np.gradient(
        U2,
        Tau,
        axis=0,
        edge_order=2
    )



    # ======================================================
    # R1
    # ======================================================


    R1=(

        dU1_dtau

        -
        (
        D1_th[None,:]*dU1_dA
        +
        D2_th[None,:]*d2U1_dA2
        )

        -
        D1_th[None,:]

    )



    # ======================================================
    # R2
    # ======================================================


    R2=(

        dU2_dtau

        -
        (
        D1_th[None,:]*dU2_dA
        +
        D2_th[None,:]*d2U2_dA2
        )

        -
        D1_th[None,:]*U1

        -
        2*D2_th[None,:]*dU1_dA

        -
        D2_th[None,:]

    )



    # ======================================================
    # 去除边界
    # 与PhysicsInformedLoss一致
    # ======================================================


    R1_inner=R1[1:-1,1:-1]

    R2_inner=R2[1:-1,1:-1]



    # ======================================================
    # 有效值过滤
    # ======================================================


    R1_valid=R1_inner[
        np.isfinite(R1_inner)
    ]


    R2_valid=R2_inner[
        np.isfinite(R2_inner)
    ]



    if len(R1_valid)==0:
        raise ValueError(
            "R1全部无效"
        )


    if len(R2_valid)==0:
        raise ValueError(
            "R2全部无效"
        )



    # ======================================================
    # RMSE
    # ======================================================


    R1_RMSE=np.sqrt(
        np.mean(
            R1_valid**2
        )
    )


    R2_RMSE=np.sqrt(
        np.mean(
            R2_valid**2
        )
    )


    Total=np.sqrt(
        R1_RMSE**2+
        R2_RMSE**2
    )



    summary={


        "R1_RMSE":R1_RMSE,


        "R2_RMSE":R2_RMSE,


        "Total_residual":Total,


        "R1_valid_ratio":
        len(R1_valid)/R1_inner.size,


        "R2_valid_ratio":
        len(R2_valid)/R2_inner.size,


        "Max_abs_R1":
        np.max(np.abs(R1_valid)),


        "Max_abs_R2":
        np.max(np.abs(R2_valid)),


        "tau_points":n_tau,


        "A_points":n_A

    }



    # ======================================================
    # 详细输出
    # ======================================================


    detail=[]


    for i,tau in enumerate(Tau):

        for j,a in enumerate(A):


            r1=R1[i,j]

            r2=R2[i,j]


            detail.append({

                "tau":tau,

                "A":a,

                "R1":r1,

                "R2":r2,


                "Total_residual":
                np.sqrt(
                    r1**2+r2**2
                ),


                "valid":
                np.isfinite(r1)
                and np.isfinite(r2)

            })


    return summary,pd.DataFrame(detail)




# ==========================================================
# 主程序
# ==========================================================


results=[]


for folder in os.listdir(INPUT_ROOT_DIR):


    folder_path=os.path.join(
        INPUT_ROOT_DIR,
        folder
    )


    if not os.path.isdir(folder_path):
        continue



    csv_path=os.path.join(
        folder_path,
        CSV_NAME
    )


    if not os.path.exists(csv_path):

        continue



    print(
        "Processing:",
        folder
    )


    try:


        summary,detail=calculate_physics_residual(
            csv_path
        )


        params=parse_folder_params(folder)


        if params:

            summary["nu"]=params[0]
            summary["kappa"]=params[1]
            summary["D"]=params[2]


        summary["folder"]=folder


        results.append(summary)



        out=os.path.join(
            OUTPUT_ROOT_DIR,
            folder
        )


        os.makedirs(
            out,
            exist_ok=True
        )


        detail.to_csv(
            os.path.join(
                out,
                "physics_residual_detail.csv"
            ),
            index=False
        )



    except Exception as e:

        print(
            "ERROR:",
            folder,
            e
        )



# ==========================================================
# 保存汇总
# ==========================================================


summary_df=pd.DataFrame(results)


summary_df.to_csv(
    os.path.join(
        OUTPUT_ROOT_DIR,
        SUMMARY_NAME
    ),
    index=False
)



print("\nFinished")
print(
    summary_df
)