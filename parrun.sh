start=`date +%s`
python python/min_en_place.py small
parallel --nice 5 --jobs 10 --colsep ' ' --will-cite -a sim_args.txt python python/ada_sim.py
python python/plot_ada_sim.py synthetic random
python python/plot_ada_sim.py synthetic group
python python/plot_ada_sim.py synthetic hier2nd
python python/plot_ada_sim.py uci hier2nd
end=`date +%s`
runtime=$((end-start))
echo $runtime
