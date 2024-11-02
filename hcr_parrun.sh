start=`date +%s`
python python/mis_en_place.py hcr
parallel --nice 5 --jobs 10 --colsep ' ' --will-cite -a hcr_zinb_args.txt python python/neo_hcr.py
python python/plot_hcr.py 
end=`date +%s`
runtime=$((end-start))
echo $runtime
