#并行执行
python train_dtc_2d.py --exp semi/SEG_addDDR_5_od_not-rim --od_rim False --device 0 &
python train_dtc_2d.py --exp semi/SEG_addDDR_5_od_rim --od_rim True --device 1 &

wait

#等待上面两个并行执行结束后，再并行执行
python train_dtc_2d.py --exp semi/SEG_addDDR_5_od_not-rim_oc-label-2 --oc_label 2 --od_rim False --device 0 &
python train_dtc_2d.py --exp semi/SEG_addDDR_5_od_not-rim_oc-label-1 --oc_label 1 --od_rim False --device 1


python train_dtc_2d.py --exp semi/SEG_addDDR_5_od_rim_outchannel-minus1 --outchannel_minus1 True --od_rim True --device 0

python train_dtc_2d.py --exp semi/SEG_addDDR_5_od_rim_outchannel-minus1_oc-label1 --oc_label 1 --outchannel_minus1 True --od_rim True --device 1

python train_dtc_2d.py --exp semi/SEG_addDDR_5_od_not-rim_outchannel-minus1_oc-label1 --oc_label 1 --device 0

python train_dtc_2d.py --exp semi/SEG_addDDR_5_od_rim_outchannel-minus1_oc-label1 --oc_label 1 --od_rim True --device 1



# supervised
python train_supervised_2d.py  --device 1