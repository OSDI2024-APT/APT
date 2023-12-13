master_addr=172.31.40.2
configs_name_all=(papers_w16_metis)
# configs_name_all=(products_w32_metis)
logs_dir=./logs/ap/new_machines2.csv


for configs in ${configs_name_all[@]}
do
  for nproc_per_node in 4
  do
    world_size=$(($nproc_per_node*4))
    configs_path=./npc_dataset_acc2/${configs}/configs.json
    cache_mode=dryrun
    # cache_mode=none

    python examples/mp_runner.py --world_size ${world_size} --nproc_per_node=${nproc_per_node} --node_rank=$1  --master_addr=${master_addr} --master_port=12345 --logs_dir ${logs_dir} --configs_path ${configs_path} --cache_mode ${cache_mode} --debug --num_epochs 50
  done
done
