import argparse
import os

from transformers.utils.QC_data_preprocessing import get_meta_info

def main(args):
    data_file = args.data_file
    out_file_path = args.out_file_path
    mode = args.mode

    ner = args.ner
    use_agent_id = args.use_agent_id
    use_turn_num = args.use_turn_num
    use_topic_object = args.use_topic_object
    agents_to_use = args.agents_to_use.strip().split() if args.agents_to_use else []

    # process data_file
    processed_chats = get_meta_info(
        data_file,
        type=mode,
        ner=ner,
        tokenize=False,
        use_agent_id=use_agent_id,
        use_turn_num=use_turn_num,
        use_topic_object=use_topic_object,
        agents_to_use=agents_to_use)

    # dump on file
    out_file_path = os.path.splitext(out_file_path)[0]
    out_file_source_path, out_file_target_path = out_file_path + '.source', out_file_path + '.target'
    with open(out_file_source_path, 'w', encoding='utf-8') as s, open(out_file_target_path, 'w', encoding='utf-8') as t:
        for sample in processed_chats:
            s.write(sample['chat'] + '\n')
            t.write(sample['target'] + '\n')
    print(f'generated {out_file_source_path} and {out_file_target_path}')
    
    return

if __name__ == "__main__":
    # Ppreprocessing from QuichChat data
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="the .json (or .json.gz) file to be processed",
    )
    parser.add_argument(
        "--out_file_path",
        type=str,
        required=True,
        help="is the name of the out file. ",
    )
    parser.add_argument('--mode', type=str, default='train_to_test',
                    choices=['train_to_test', 'test'],
                    help='the QuickChat file type. train_to_test means that the input is a train file, but the chats must be splitted into contexts and targets.')
    parser.add_argument('--ner', action='store_true',
                    help='use ner utterances from json files (both in training and test for LM)')
    parser.add_argument('--use_agent_id', action='store_true',
                    help='use agent_id augmentation in text')
    parser.add_argument('--use_turn_num', action='store_true',
                    help='use turn_num augmentation in text')
    parser.add_argument('--use_topic_object', action='store_true',
                    help='use topic_object augmentation in text. (i.e. just the #object in topic)')
    parser.add_argument('--agents_to_use', type=str, default=None,
                    help='list of agents id divided by space.')
    args = parser.parse_args()
    main(args)


