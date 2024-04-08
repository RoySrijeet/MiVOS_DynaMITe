import pandas as pd
import numpy as np
from collections import defaultdict

def summarize_results(results):
    summary = defaultdict()
    summary['meta'] = {}
    summary['meta']['iou_threshold'] = results['iou_threshold']
    summary['meta']['iou_checkpoints'] = results['iou_checkpoints']
    summary['meta']['max_interactions_per_frame'] = results['max_interactions']
    summary['meta']['max_rounds_per_sequence'] = results['max_rounds']

    summary['meta']['total_interactions_over_dataset'] = results['total_num_interactions'][0]
    all_interactions = results['all_interactions']
    all_interactions_per_instance = results['all_interactions_per_instance']

    summary['meta']['total_rounds_over_dataset'] = results['total_num_rounds'][0]
    all_rounds = results['all_rounds']
    all_interactions_per_round = results['all_interactions_per_round']
    
    all_instance_level_iou = results['all_instance_level_iou']
    all_j_and_f = results['all_j_and_f']
    all_jaccard = results['all_jaccard']
    all_contour = results['all_contour']
    all_ious = results['all_ious']

    
    avg_iou_over_dataset = []
    avg_jandf_over_dataset = []
    total_failed_instances = []
    total_failed_frames = []
    total_failed_sequences = 0

    total_frames_over_dataset = []
    total_frames_interacted = []
    total_instances_over_dataset = []
    total_instances_interacted = []
    total_background_clicks = 0
    total_foreground_clicks = 0
    
    round_results = []

    seqs = list(all_ious.keys())
    for seq in seqs:
        summary[seq] = {}
        ious = all_ious[seq]

        # metrics
        summary[seq]['max_IoU'] = max(ious)
        summary[seq]['min_IoU'] = min(ious)
        summary[seq]['avg_IoU'] = sum(ious)/len(ious)

        summary[seq]['max_J'] = max(all_jaccard[seq])
        summary[seq]['min_J'] = min(all_jaccard[seq])
        summary[seq]['avg_J'] = sum(all_jaccard[seq])/len(all_jaccard[seq])

        summary[seq]['max_F'] = max(all_contour[seq])
        summary[seq]['min_F'] = min(all_contour[seq])
        summary[seq]['avg_F'] = sum(all_contour[seq])/len(all_contour[seq])

        summary[seq]['avg_J_AND_F'] = sum(all_j_and_f[seq])/len(all_j_and_f[seq])
        
        avg_iou_over_dataset.append(summary[seq]['avg_IoU'])
        avg_jandf_over_dataset.append(summary[seq]['avg_J_AND_F'])
        
        # failed sequences, frames, instances
        total_failed_frames.append(sum(1 for i in ious if i < results['iou_threshold']))
        if summary[seq]['avg_IoU'] < results['iou_threshold']:
            total_failed_sequences +=1
        
        instance_level_iou = all_instance_level_iou[seq]
        failed_instances = 0
        for ious in instance_level_iou:
            if len(ious) !=0:
                for iou in ious:
                    if iou < results['iou_threshold']:
                        failed_instances += 1
        total_failed_instances.append(failed_instances)

        interactions = all_interactions[seq]    

        summary[seq]['total_frames'] = len(interactions)
        total_frames_over_dataset.append(summary[seq]['total_frames'])
        summary[seq]['frames_interacted'] = np.count_nonzero(np.array(interactions))
        total_frames_interacted.append(summary[seq]['frames_interacted'])
        summary[seq]['total_interactions'] = sum(interactions)
        summary[seq]['num_of_rounds'] = all_rounds[seq]

        object_clicks = defaultdict(lambda:0)
        for clicks in all_interactions_per_instance[seq]:
            if len(clicks) !=0:
                for c in range(len(clicks)):      # last click for bg
                    if c==len(clicks)-1:  # bg click
                        total_background_clicks += clicks[c]
                    else:
                        total_foreground_clicks += clicks[c]
                        object_clicks[c] += clicks[c]
        summary[seq]['instance_wise_interactions'] = list(object_clicks.items())
        total_instances_over_dataset.append(len(list(object_clicks.keys())))
        total_instances_interacted.append(np.count_nonzero(np.array(list(object_clicks.values()))))
        
        for item in all_interactions_per_round[seq]:
            round_results.append([seq] + item)

    df = pd.DataFrame(round_results, columns=['sequence', 'round', 'dynamite_loop', 'frame_idx', 'object_idx', 'num_interactions', 'frame_avg_iou', 'seq_avg_iou', 'seq_avg_j_and_f' ])

    summary['meta']['total_foreground_interactions_over_dataset'] = total_foreground_clicks
    summary['meta']['total_background_interactions_over_dataset'] = total_background_clicks
    
    summary['meta']['avg_iou_over_dataset'] = sum(avg_iou_over_dataset)/len(avg_iou_over_dataset)
    summary['meta']['avg_jandf_over_dataset'] = sum(avg_jandf_over_dataset)/len(avg_jandf_over_dataset)
    
    summary['meta']['total_frames_over_dataset'] = sum(total_frames_over_dataset)
    summary['meta']['total_frames_interacted'] = sum(total_frames_interacted)
    
    summary['meta']['total_instances_over_dataset'] = sum(total_instances_over_dataset)
    summary['meta']['total_instances_interacted'] = sum(total_instances_interacted)
    
    summary['meta']['total_failed_sequences'] = total_failed_sequences
    summary['meta']['total_failed_frames'] = sum(total_failed_frames)    
    summary['meta']['total_failed_instances'] = sum(total_failed_instances)

    return summary,df

def summarize_round_results(df, iou_threshold):
    table = []
    sequences = set(df['sequence'])
    for seq in sequences:
        entry = [seq]
        df_seq = df[df['sequence']==seq].reset_index(drop=True)
        
        # num instances
        num_instances = len(df_seq['object_idx'][0])
        entry.append(num_instances)
        
        # num interactions
        num_interactions = list(df_seq['num_interactions'])[-1]
        entry.append(num_interactions)  
        
        # num rounds
        num_rounds = list(df_seq['round'])[-1]
        entry.append(num_rounds)

        # IoU checkpoints
        entry.append(iou_threshold)
        checkpoints = [0.85, 0.90, 0.95, 0.99]

        # IoU
        frame_avg_iou = []
        frame_avg_iou_ = list(df_seq['frame_avg_iou'])
        first_frame_final_iou = 0
        for idx,val in enumerate(frame_avg_iou_):
            if val != '-':
                frame_avg_iou.append(float(val))
            else:
                if df_seq['frame_idx'][idx-1]=='0':
                    first_frame_final_iou = eval(df_seq['frame_avg_iou'][idx-1])

        
        max_iou = max(frame_avg_iou)
        max_idx = df_seq['num_interactions'][frame_avg_iou.index(max_iou)]
        for idx, iou in enumerate(frame_avg_iou):
            while float(iou)>=checkpoints[0]:
                t = checkpoints.pop(0)
                entry.append(df_seq['num_interactions'][idx])
        for c in checkpoints:
            entry.append(0)
        entry.append(first_frame_final_iou)     # IoU of first frame
        entry.append([round(max_iou,6), max_idx])           # max IoU reached
        entry.append(float(list(df_seq['seq_avg_iou'])[-1]))
        entry.append(float(list(df_seq['seq_avg_j_and_f'])[-1]))
        table.append(entry)
    
    table_df = pd.DataFrame(table, columns=['sequence', 'num_instances', 'num_interactions',  'num_rounds', 'iou_threshold', 'iou_0.85', 'iou_0.90', 'iou_0.95', 'iou_0.99', 'first_frame_final_iou', '[max_iou, idx]', 'seq_avg_iou', 'seq_avg_jandf'])
    final_entry = ['TOTAL']
    final_entry.append(table_df['num_instances'].sum())
    final_entry.append(table_df['num_interactions'].sum())
    final_entry.append(table_df['num_rounds'].sum())
    final_entry.append(iou_threshold)
    final_entry.append(np.count_nonzero(table_df['iou_0.85']))
    final_entry.append(np.count_nonzero(table_df['iou_0.90']))
    final_entry.append(np.count_nonzero(table_df['iou_0.95']))
    final_entry.append(np.count_nonzero(table_df['iou_0.99']))
    final_entry.append(table_df['first_frame_final_iou'].mean())
    final_entry.append('-')
    final_entry.append(table_df['seq_avg_iou'].mean())
    final_entry.append(table_df['seq_avg_jandf'].mean())
    table_df.loc[len(table_df)] = final_entry
    return table_df