import json
import gzip
import argparse
import glob
import re
import os
import shutil

AGENT_TOKEN, CLOSE_AGENT_TOKEN = '<AGENT>', '</AGENT>'
USER_TOKEN, CLOSE_USER_TOKEN = '<USER>', '</USER>'

__pattern = re.compile('\w+|[^\w\s]')
def tokenize(text, pattern=__pattern):
    return __pattern.findall(text)

class TurnFromDataClean:
    def __init__(self, raw_turn, ner=False, tokenize=False, use_agent_id=False, use_turn_num=False, use_topic_object=False, agents_to_use=None):
        self.raw_turn = raw_turn
        self.ner = ner # if a ner utterance is used
        self.use_agent_id = use_agent_id
        self.use_turn_num = use_turn_num
        self.use_topic_object = use_topic_object
        self.agents_to_use = agents_to_use
        self.utterance_no = raw_turn['utterance_no']
        self.speaker_id = raw_turn['speaker_id']
        self.sender_id = self.get_agent_to_use()
        self.topic_object = raw_turn['topic_object'] if raw_turn.get('topic_object') else ''
        if self.ner:
            self.utterance = self._tokenize_utterance(raw_turn['ner']['utterance']) if tokenize else raw_turn['ner']['utterance']
        else:
            self.utterance = self._tokenize_utterance(raw_turn['utterance']) if tokenize else raw_turn['utterance']

    def get_agent_to_use(self):
        """ if sender_id is among the agent to use then keep it, otherwise delete it"""
        sender_id = self.raw_turn['sender_id'] if self.raw_turn.get('sender_id') else ''  # null or agent_id
        if self.agents_to_use and sender_id != '':
            if sender_id not in self.agents_to_use:
                sender_id = ''
        return sender_id

    def _tokenize_utterance(self, utterance):
        return ' '.join(tokenize(utterance))

    def format(self):
        start_speaker_token = AGENT_TOKEN if self.speaker_id == 'AGENT' else USER_TOKEN
        end_speaker_token = CLOSE_AGENT_TOKEN if self.speaker_id == 'AGENT' else CLOSE_USER_TOKEN
        add_pipe = self.use_agent_id or self.use_topic_object or self.use_turn_num
        sent = f'{start_speaker_token} '
        if self.use_turn_num:
            sent += f'{self.utterance_no} '
        if self.use_agent_id and self.sender_id != '':
            sent += f'{self.sender_id} '
        if self.use_topic_object and self.topic_object != '':
            sent += f'{self.topic_object} '
        if add_pipe:
            sent += f'| '
        sent += f'{self.utterance} {end_speaker_token}'
        return sent

    def format_meta(self):
        assert self.speaker_id == 'AGENT'
        add_pipe = self.use_agent_id or self.use_topic_object or self.use_turn_num
        sent = f'{AGENT_TOKEN} '
        if self.use_turn_num:
            sent += f'{self.utterance_no} '
        if self.use_agent_id and self.sender_id != '':
            sent += f'{self.sender_id} '
        if self.use_topic_object and self.topic_object != '':
            sent += f'{self.topic_object} '
        if add_pipe:
            sent += f'|'
        return sent

class ChatFromDataClean:
    def __init__(self, agents, turns, chat_id, type='train', ner=True, tokenize=False, use_agent_id=False, use_turn_num=False, use_topic_object=False, agents_to_use=None):
        self.ner = ner
        self.use_agent_id = use_agent_id
        self.use_turn_num = use_turn_num
        self.use_topic_object = use_topic_object
        self.agents_to_use = agents_to_use
        self.raw_agents_info = agents
        self.raw_turns = turns
        self.tokenize = tokenize
        self.chat_id = chat_id

        self.agent_id = self._get_agent_id()
        self.agent_alias = self._get_agent_alias()
        self.agent_group = self._get_agent_group()
        self.business_unit = self._get_business_unit()
        self.turns = self._get_turns()
        self.agent_exists = self.check_agent_exists()
        self.turns_till_last_agent = self._turns_till_last_agent() if (type == 'train-to-test' and self.agent_exists) else None

    def check_agent_exists(self):
        agent_exists = False
        for ix, t in enumerate(self.turns):
            if t.speaker_id == 'AGENT':
                agent_exists = True
                break
        return agent_exists

    def _get_agent_alias(self):
        agent_alias = self.raw_agents_info[0].get("agent_alias") # in some cases there is no alias
        return agent_alias

    def _get_agent_id(self):
        agent_id = self.raw_agents_info[0]["agent_id"]
        return agent_id

    def _get_agent_group(self):
        agent_group = self.raw_agents_info[0]["agent_group"]
        return agent_group

    def _get_business_unit(self):
        business_unit = self.raw_agents_info[0]["business_unit"]
        return business_unit

    def _get_turns(self):
        turns = []
        for raw_turn in self.raw_turns:
            turn = TurnFromDataClean(raw_turn, self.ner, self.tokenize, self.use_agent_id, self.use_turn_num, self.use_topic_object, self.agents_to_use)
            turns.append(turn)
        return turns

    def format(self):
        chat_to_string = ''
        for t in self.turns:
            chat_to_string += t.format() + ' '
        return chat_to_string.strip()

    def format_as_test(self):
        # get rid of last agent turn and add meta info at the end
        chat_to_string = ''
        assert self.turns_till_last_agent[-1].speaker_id == 'AGENT'
        for t in self.turns_till_last_agent[:-1]:
            chat_to_string += t.format() + ' '
        chat_to_string += self.turns_till_last_agent[-1].format_meta()
        return chat_to_string.strip()

    def get_target(self):
        assert self.turns_till_last_agent[-1].speaker_id == 'AGENT'
        return self.turns_till_last_agent[-1].utterance

    def _turns_till_last_agent(self):
        index = None
        for ix, t in enumerate(self.turns[::-1]):
            if t.speaker_id == 'AGENT':
                index = ix
                break
        assert index is not None
        if ix == 0:
            return self.turns
        else:
            return self.turns[:-ix]

class TurnFromMeta:
    def __init__(self, utterance, utterance_no, speaker_id, sender_id, topic_object=None, agent_group=None, business_unit=None, utterance_type=''):
        self.utterance = utterance
        self.utterance_no = utterance_no
        self.speaker_id = speaker_id
        self.sender_id = sender_id
        self.topic_object = topic_object
        self.agent_group = agent_group
        self.business_unit = business_unit
        self.utterance_type = utterance_type

    def format(self, concat_agent_id=False, concat_turn_n=False, concat_agent_group=False, concat_business_unit=False, concat_topic_object=False, max_len=246):
        token = AGENT_TOKEN if self.speaker_id == 'AGENT' else USER_TOKEN
        close_token = CLOSE_AGENT_TOKEN if self.speaker_id == 'AGENT' else CLOSE_USER_TOKEN

        if self.speaker_id == 'USER':
            concat_agent_id = concat_agent_id
            concat_agent_group = False
            concat_business_unit = False
            concat_turn_n = concat_turn_n
            concat_topic_object = concat_topic_object

        #turn_formatted = ' '.join(self.utterance[:max_len])
        if not (concat_agent_id or concat_topic_object or concat_turn_n):
            uttr = f'{token} {self.utterance.strip()} {close_token} '
        else:
            turn_formatted = '| ' + self.utterance.strip()
            if concat_topic_object and self.topic_object:
                turn_formatted = self.topic_object + ' ' + turn_formatted
            if concat_agent_id and self.sender_id:
                turn_formatted = self.sender_id + ' ' + turn_formatted
            if concat_agent_group and self.agent_group:
                for ag in self.agent_group:
                    turn_formatted = ag + ' ' + turn_formatted
            if concat_business_unit and self.business_unit:
                for bu in self.business_unit:
                    turn_formatted = bu + ' ' + turn_formatted
            if concat_turn_n and self.utterance_no:
                turn_formatted = str(self.utterance_no) + ' ' + turn_formatted
            uttr = token + ' ' + turn_formatted.strip() + ' ' + close_token + ' '
        return uttr

    def format_meta(self, concat_agent_id=True, concat_turn_n=True, concat_topic_object=False):
        meta_string = AGENT_TOKEN + ' '
        if concat_turn_n:
            meta_string += str(self.utterance_no) + ' '
        if concat_agent_id and self.sender_id:
            meta_string += self.sender_id + ' '
        if concat_topic_object and self.topic_object:
            meta_string += self.topic_object + ' '
        if concat_turn_n or concat_agent_id or concat_topic_object:
            meta_string += '|'
        return meta_string

class ChatFromMeta:
    def __init__(self, question_id, question, candidates, agent_info, label, reference_answer, turn_num, topic_object, use_agent_id=False, use_turn_num=False, use_topic_object=False, agents_to_use=None, use_agent_group=False, use_business_unit=False):
        self.question_id = question_id
        self.question = question
        self.candidates = candidates
        self.label = label[0] if label else None
        self.reference_answer = reference_answer
        self.agent_info = agent_info
        self.turn_num = turn_num
        self.topic_object = topic_object
        self.use_agent_id = use_agent_id
        self.use_turn_num = use_turn_num
        self.use_topic_object = use_topic_object
        self.agents_to_use = agents_to_use
        self.use_agent_group = use_agent_group
        self.use_business_unit = use_business_unit
        self.turns = self._get_turns()
        self.target = self._get_target()

    def is_out_of_pool(self):
        if self.label:
            return False
        return True

    def _set_last_turn_num(self, turns):
        # set last agent turn with turn num,
        # if there is a user turn after last agent one, use turn_num -1
        user_turn = False
        for t in turns[::-1]:
            if t.speaker_id == 'AGENT':
                if user_turn == False:
                    t.utterance_no = self.turn_num
                else:
                    t.utterance_no = self.turn_num -1
                break
            else:
                user_turn = True
        return turns

    def get_agent_alias(self):
        return self.agent_info['agent_alias']

    def get_agent_id(self):
        agent_id = self.agent_info['agent_id']
        if agent_id and self.agents_to_use:
            if agent_id not in self.agents_to_use:
                agent_id = None
        return agent_id

    def get_agent_group(self):
        return self.agent_info['agent_group']

    def get_business_unit(self):
        return self.agent_info['business_unit']

    def _get_turns(self):
        chat = self.question.strip('<CHAT>|</CHAT>').strip()
        matches = re.findall(r'(<AGENT>(.*?)</AGENT>|<USER>(.*?)</USER>)', chat)
        turns = []
        agent_turn_num, user_augmented = 1, False # set to None since you want the firsts agent turns till user to have no
        user_turn_num, agent_augmented = 0, False
        usr_counter, agent_counter = 0, 0
        for m in matches:
            if m[1] == '':
                usr_counter += 1
                turns.append(TurnFromMeta(m[2], user_turn_num, 'USER', None, self.topic_object))  # (list, 'USER')
                if not agent_augmented:
                    agent_turn_num += 1
                    agent_augmented = True
                    user_augmented = False
            else:
                agent_counter +=1
                if agent_turn_num == 1 and usr_counter == 0:
                    # add agent id if we have already seen a usr utterance. otherwise are probably still initial bot info
                    turns.append(TurnFromMeta(m[1], agent_turn_num, 'AGENT', None, self.topic_object, [self.get_agent_group()], [self.get_business_unit()]))
                else:
                    turns.append(TurnFromMeta(m[1], agent_turn_num, 'AGENT', self.get_agent_id(), self.topic_object, [self.get_agent_group()], [self.get_business_unit()]))

                if not user_augmented:
                    user_turn_num += 1
                    user_augmented = True
                    agent_augmented = False

        return turns

    def format_chat(self, add_agent_token=True, cls_mode=False, train_mode_LM=False, crop=None, end_meta=False):
        natural_turns_uttr = [t.format(self.use_agent_id, self.use_turn_num, self.use_agent_group, self.use_business_unit, self.use_topic_object) for t in self.turns]
        if end_meta:
            # add dummy turn just to get meta info at the end
            natural_turns_uttr += TurnFromMeta('',self.turn_num,'AGENT',self.get_agent_id(), self.topic_object, None).format_meta(concat_agent_id=self.use_agent_id, concat_turn_n=self.use_turn_num, concat_topic_object=self.use_topic_object)

        chat = '<eos> ' if not (cls_mode or train_mode_LM) else ''
        for t in natural_turns_uttr:
            chat += t
        if add_agent_token and not (cls_mode or train_mode_LM):
            chat += '<AGENT>'
        return chat

    def _get_target(self):
        target = None
        if self.label is not None:
            for c in self.candidates:
                if c['id'] == self.label:
                    target = c['answer'][0]
                    break
        else:
            target = self.reference_answer[0]['answer']

        assert target is not None
        return target

def load_dataclean_chat(line, type, ner,tokenize=False, use_agent_id=False, use_turn_num=False, use_topic_object=False, agents_to_use=None):
    """
    return a Chat from a json string. if ner, then use the "ner" entry in json as the utterance.
    Tokenize turns if required. (for BPE vocab is not required)
    """
    sample = json.loads(line)
    # if train.json will have agents, if test.json will have agent_info

    chat = ChatFromDataClean(sample.get('agents'),
                sample.get('turns'),
                sample.get('chat_id'),
                type,
                ner,
                tokenize=tokenize,
                use_agent_id=use_agent_id,
                use_turn_num=use_turn_num,
                use_topic_object=use_topic_object,
                agents_to_use=agents_to_use)
    return chat

def load_meta_chat(line, use_agent_id=False, use_turn_num=False, use_topic_object=False, agents_to_use=None, use_agent_group=False, use_business_unit=False):
    sample = json.loads(line)
    assert 'question' in sample
    assert 'candidates' in sample
    assert 'agent_info' in sample
    if 'label' not in sample:
        assert 'reference-answer' in sample

    chat = ChatFromMeta(sample.get('question_id'),
                sample.get('question'),
                sample.get('candidates'),
                sample.get('agent_info'),
                sample.get('label'),
                sample.get('reference-answer'),
                sample.get('turn_num'),
                sample.get('topic_object'),
                use_agent_id=use_agent_id,
                use_turn_num=use_turn_num,
                use_topic_object=use_topic_object,
                agents_to_use=agents_to_use,
                use_agent_group=use_agent_group,
                use_business_unit=use_business_unit)
    return chat

def get_sub_dir_file_paths(dataset_root, extension):
    """
    return a list of all the file paths which terminate with the provided extension,
    and are within dataset_root sub-directories. i.e dataset_root=data_clean/ -> sub-dirs=[201811/, 201812/]
    """
    path_files = []
    sub_dirs = [os.path.join(dataset_root, dI) for dI in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, dI))]
    sub_dirs.sort()
    for sub_d in sub_dirs:
        sub_d_files = glob.glob(os.path.join(sub_d, f'*{extension}'))
        path_files += sub_d_files
    return path_files

def decompress_gzip_file(path):
    file_name = os.path.splitext(path)[0]  # get file name
    with gzip.open(path, 'rb') as f_in, open(file_name, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return

def get_meta_info(json_file, type='train', ner=False, tokenize=False, use_agent_id=False, use_turn_num=False, use_topic_object=False, agents_to_use=None):
    meta_info = []
    if os.path.splitext(json_file)[-1] == '.gz':
        f = gzip.open(json_file, 'rb')
        lines = f.readlines()
        lines = [l.decode('utf-8') for l in lines]
    else:
        f = open(json_file, 'r', encoding='utf-8')
        lines = f.readlines()
    for ix, line in enumerate(lines):
        meta = {}
        if type == 'train':
            chat = load_dataclean_chat(line, 'train', ner, tokenize, use_agent_id, use_turn_num, use_topic_object, agents_to_use)
            if chat.agent_exists:  # chat can be None if there are errors in the json file (like there is no agent)
                meta['question_id'] = chat.chat_id
                meta['chat'] = chat.format()
                meta['agent_id'] = chat.agent_id
                meta['agent_alias'] = chat.agent_alias
                meta['agent_group'] = chat.agent_group
                meta['business_unit'] = chat.business_unit
                meta_info.append(meta)
        elif type == 'train_to_test':
            chat = load_dataclean_chat(line, 'train-to-test', ner, tokenize, use_agent_id, use_turn_num, use_topic_object, agents_to_use)
            if chat.agent_exists:
                meta['question_id'] = chat.chat_id
                meta['chat'] = chat.format_as_test()
                meta['agent_id'] = chat.agent_id
                meta['agent_alias'] = chat.agent_alias
                meta['agent_group'] = chat.agent_group
                meta['business_unit'] = chat.business_unit
                meta['target'] = chat.get_target()
                meta_info.append(meta)
        else:
            chat = load_meta_chat(line, use_agent_id, use_turn_num, use_topic_object, agents_to_use, use_agent_group=False, use_business_unit=False)
            meta['chat'] = chat.format_chat(add_agent_token=False, cls_mode=True, train_mode_LM=False, crop=None, end_meta=True) # cls_mode=True fast way to get NO <eos> at the begining.
            meta['agent_id'] = chat.get_agent_id()
            meta['agent_alias'] = chat.get_agent_alias()
            meta['agent_group'] = chat.get_agent_group()
            meta['business_unit'] = chat.get_business_unit()
            meta['question_id'] = chat.question_id
            meta['target'] = chat.target
            meta_info.append(meta)
    return meta_info

def main(args):
    #train test
    m1 = get_meta_info('/gpfs/ess1_fs1/nlu/data/users/giovanni_bonetta/transformerxl/data/LM/Sprint_6Ms/LM_datasets/train/2020-01_0.json.gz',
                       type='train',
                       ner=False,
                       tokenize=False,
                       use_agent_id=True,
                       use_turn_num=True,
                       use_topic_object=False,
                       agents_to_use=None)
    m2 = get_meta_info(
        '/gpfs/ess1_fs1/nlu/data/users/giovanni_bonetta/transformerxl/data/LM/Sprint_6Ms/LM_datasets/train/2020-01_0.json.gz',
        type='train',
        ner=False,
        tokenize=False,
        use_agent_id=True,
        use_turn_num=True,
        use_topic_object=False,
        agents_to_use=['oo408068'])

    m3 = get_meta_info(
        '/gpfs/ess1_fs1/nlu/data/users/giovanni_bonetta/transformerxl/data/LM/Sprint_6Ms/LM_datasets/train/2020-01_0.json.gz',
        type='train',
        ner=False,
        tokenize=False,
        use_agent_id=True,
        use_turn_num=False,
        use_topic_object=False,
        agents_to_use=['oo408068'])

    m4 = get_meta_info(
        '/gpfs/ess1_fs1/nlu/data/users/giovanni_bonetta/transformerxl/data/LM/Sprint_6Ms/LM_datasets/train/2020-01_0.json.gz',
        type='train',
        ner=False,
        tokenize=False,
        use_agent_id=True,
        use_turn_num=True,
        use_topic_object=True,
        agents_to_use=['oo408068'])

    mb = get_meta_info('/gpfs/ess1_fs1/nlu/data/users/giovanni_bonetta/transformerxl/data/LM/2020-07-01/Bofa/ner/201909/2019-09-02.json',
                       type='train',
                       ner=True,
                       tokenize=False,
                       use_agent_id=True,
                       use_turn_num=True,
                       use_topic_object=False,
                       agents_to_use=None)

    print("TRAIN")
    print(f'm1 id, num         : {m1[0]["chat"]}')
    print(f'm2 id, num, ag     : {m2[0]["chat"]}')
    print(f'm3 id, ag          : {m3[0]["chat"]}')
    print(f'm4 id, ag, num, obj: {m4[0]["chat"]}')
    print(f'm1 id, num         : {m1[1]["chat"]}')
    print(f'm2 id, num, ag     : {m2[1]["chat"]}')
    print(f'm3 id, ag          : {m3[1]["chat"]}')
    print(f'm4 id, ag, num, obj: {m4[1]["chat"]}')

    print(f'bofa id, ag, ner: {mb[1]["chat"]}')
    print(f'bofa id, ag, ner: {mb[2]["chat"]}')

    # test test
    m1 = get_meta_info(
        '/gpfs/ess1_fs1/nlu/data/users/giovanni_bonetta/transformerxl/data/LM/Sprint_6Ms/KNN_datasets/test/test_100.json',
        type='test',
        ner=False,
        tokenize=False,
        use_agent_id=True,
        use_turn_num=True,
        use_topic_object=False,
        agents_to_use=None)
    m2 = get_meta_info(
        '/gpfs/ess1_fs1/nlu/data/users/giovanni_bonetta/transformerxl/data/LM/Sprint_6Ms/KNN_datasets/test/test_100.json',
        type='test',
        ner=False,
        tokenize=False,
        use_agent_id=True,
        use_turn_num=True,
        use_topic_object=False,
        agents_to_use=['er363637'])

    m3 = get_meta_info(
        '/gpfs/ess1_fs1/nlu/data/users/giovanni_bonetta/transformerxl/data/LM/Sprint_6Ms/KNN_datasets/test/test_100.json',
        type='test',
        ner=False,
        tokenize=False,
        use_agent_id=True,
        use_turn_num=False,
        use_topic_object=False,
        agents_to_use=['er363637'])

    m4 = get_meta_info(
        '/gpfs/ess1_fs1/nlu/data/users/giovanni_bonetta/transformerxl/data/LM/Sprint_6Ms/KNN_datasets/test/test_100.json',
        type='test',
        ner=False,
        tokenize=False,
        use_agent_id=True,
        use_turn_num=True,
        use_topic_object=True,
        agents_to_use=['er363637'])

    mb = get_meta_info(
        '/gpfs/ess1_fs1/nlu/data/users/giovanni_bonetta/transformerxl/data/LM/2020-07-01/Bofa/meta2-freq/test/test_100.json',
        type='test',
        ner=True,
        tokenize=False,
        use_agent_id=True,
        use_turn_num=True,
        use_topic_object=False,
        agents_to_use=None)

    print("TEST")
    print(f'm1 id, num         : {m1[0]["chat"]}')
    print(f'm2 id, num, ag     : {m2[0]["chat"]}')
    print(f'm3 id, ag          : {m3[0]["chat"]}')
    print(f'm4 id, ag, num, obj: {m4[0]["chat"]}')
    print(f'm1 id, num         : {m1[50]["chat"]}')
    print(f'm2 id, num, ag     : {m2[50]["chat"]}')
    print(f'm3 id, ag          : {m3[50]["chat"]}')
    print(f'm4 id, ag, num, obj: {m4[50]["chat"]}')

    print(f'bofa id, ag, ner: {mb[1]["chat"]}')
    print(f'bofa id, ag, ner: {mb[2]["chat"]}')

    exit()
    """
    simple function to generate targets for BLEU computations or others.
    """
    json_file_path = args.json_file

    meta_info_test = get_meta_info(json_file_path, type='test', ner=False, tokenize=False, use_turn_num=False, use_agent_id=False, use_topic_object=False, agents_to_use=[])
    print(f'meta_info_test len: {len(meta_info_test)}')

    targets_file_path = os.path.splitext(json_file_path)[0] + '.targets'
    with open(targets_file_path, 'w', encoding='utf-8') as f:
        for i, it in enumerate(meta_info_test):
            f.write(it["target"] + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
    parser.add_argument('--json_file', type=str, default=None,
                        help='specific json file to get meta info from')
    args = parser.parse_args()
    main(args)

