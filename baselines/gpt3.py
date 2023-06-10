import json
import pathlib
from model.consts import task_names
import logging
import numpy as np
import openai
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/Users/jiang/MEWL')
parser.add_argument('--task_name', type=str, default='object')
parser.add_argument('--log_path', type=str, default='./')

class MewlCaptioner():
    def __init__(self):
        self.task_names = task_names

    def _load_info(self, path):
        episode_path = pathlib.Path(path)
        with open(episode_path / 'info.json') as f:
            info = json.load(f)
        
        return info

    def _get_name(self, obj):
        return ' '.join([obj[3], obj[1], obj[2], obj[0]])

    def _caption_objects(self, panel):
        objects = list(map(self._get_name, panel['objects']))
        scene_description = f"A {' and a '.join(objects)}."

        return scene_description

    def _caption_pragmatic_objects(self, panel):
        objects = list(map(self._get_name, panel['objects']))
        arrow_idx_to = objects[panel['arrow_idx']]
        scene_description = f"A {' and a '.join(objects)}. And a finger is pointing to the {arrow_idx_to}."

        return scene_description

    def _base_naming(self, episode_path):
        contexts = []
        info = self._load_info(episode_path)        

        for panel in info['context_panels']:
            caption = panel['name'].split(' ')
            if caption[-1][0] == '#':
                caption = caption[:-1]

            caption = ' '.join(caption)
            scene_description = self._caption_objects(panel)

            contexts.append((scene_description, caption))

        query = info['question']        
        query_description = self._caption_objects(query)
        choices = query['choices']
        answer = query['answer']

        return contexts, (query_description, choices, answer)

    def color(self, episode_path):
        return self._base_naming(episode_path)

    def material(self, episode_path):
        return self._base_naming(episode_path)

    def shape(self, episode_path):
        return self._base_naming(episode_path)
    
    def object(self, episode_path):
        return self._base_naming(episode_path)
    
    def number(self, episode_path):
        return self._base_naming(episode_path)

    def composite(self, episode_path):
        return self._base_naming(episode_path)
    
    def pragmatic(self, episode_path):
        contexts = []
        info = self._load_info(episode_path)        

        for panel in info['context_panels']:
            caption = panel['name'].split(' ')
            if caption[-1][0] == '#':
                caption = caption[:-1]

            caption = ' '.join(caption)
            scene_description = self._caption_pragmatic_objects(panel)

            contexts.append((scene_description, caption))

        query = info['question']        
        query_description = self._caption_pragmatic_objects(query)
        choices = query['choices']
        answer = query['answer']

        return contexts, (query_description, choices, answer)

    def _compute_all_relationships(self, scene_struct, eps=0.2):
        """
        Computes relationships between all pairs of objects in the scene.

        Returns a dictionary mapping string relationship names to lists of lists of
        integers, where output[rel][i] gives a list of object indices that have the
        relationship rel with object i. For example if j is in output['left'][i] then
        object j is left of object i.
        """
        all_relationships = {}
        for name, direction_vec in scene_struct['directions'].items():
            if name == 'above' or name == 'below':
                continue
            all_relationships[name] = {}
            for i, obj1 in enumerate(scene_struct['objects']):
                obj1['name'] = f"{obj1['size']} {obj1['color']} {obj1['material']} {obj1['shape']}"
                coords1 = obj1['3d_coords']
                related = set()
                for j, obj2 in enumerate(scene_struct['objects']):
                    obj2['name'] = f"{obj2['size']} {obj2['color']} {obj2['material']} {obj2['shape']}"
                    if obj1 == obj2:
                        continue
                    coords2 = obj2['3d_coords']
                    diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                    dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                    if dot > eps:
                        related.add(obj2['name'])
                all_relationships[name][obj1['name']] = list(related)
        return all_relationships


    def _parse_spatial_relation(self, episode_path, name):
        captions = []

        with open(pathlib.Path(episode_path) / f'{name}.json') as f:
            scene_info = json.load(f)
            all_relationships = self._compute_all_relationships(scene_info)
            # from front to behind
            f2b = all_relationships['behind']       
            f2b_order = [None, None, None]
            for sub, objs in f2b.items():
                if len(objs) == 1:
                    f2b_order[1] = sub
                    f2b_order[2] = objs[0]
                if len(objs) == 2:
                    f2b_order[0] = sub

            captions.append(f"The {f2b_order[1]} is in front of the {f2b_order[2]} and behind the {f2b_order[0]}.")

            # from left to right
            l2r = all_relationships['right']
            l2r_order = [None, None, None]
            for sub, objs in l2r.items():
                if len(objs) == 1:
                    l2r_order[1] = sub
                    l2r_order[2] = objs[0]
                if len(objs) == 2:
                    l2r_order[0] = sub                
            
            captions.append(f"The {l2r_order[1]} is on the left of the {l2r_order[2]} and on the right of the {l2r_order[0]}.")

        return " ".join(captions)

    def _spatial_naming(self, episode_path):
        contexts = []
        info = self._load_info(episode_path)        

        for panel in info['context_panels']:
            caption = panel['name']
            scene_description = self._parse_spatial_relation(episode_path, caption)        

            contexts.append((scene_description, caption))

    
        query = info['question']        
        query_description = self._parse_spatial_relation(episode_path, 'question')
        choices = query['choices']
        answer = query['answer']

        return contexts, (query_description, choices, answer)

    def bootstarp(self, episode_path):
        return self._spatial_naming(episode_path)

    def relation(self, episode_path):
        return self._spatial_naming(episode_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test_item(inputs, label, gpt3):
    logits = []
    for query in inputs:
        response = gpt3.create(engine="text-davinci-003",
                        prompt=query,
                        temperature=0.0,
                        max_tokens=0,
                        echo=True,
                        logprobs=0)
        logits.append(sum(response['choices'][0]['logprobs']['token_logprobs'][1:]))

    logits = np.array(logits)
    print(logits)
    pred = np.argmax(logits)
    return pred

def test_task(captioner, dataset_path, task_name, logger, average_meter, gpt3):
    dataset_path = pathlib.Path(dataset_path) / task_name
    all_episodes = dataset_path.glob('*/')
    all_episodes = sorted(all_episodes, key=lambda x: int(x.name))

    caption_func = getattr(captioner, task_name)

    for i, epi in enumerate(all_episodes):
        prompt = "Please name the target object according to the above context.\n\n"
        contexts, query = caption_func(epi)
        for context in contexts:
            prompt += f"Context: {context[0]}\nName: {context[1]}\n\n"

        prompt += f"Context: {query[0]}\nName: "
        query_description, choices, answer = query

        inputs = []
        for choice in choices:
            inputs.append(prompt + choice)

        label = choices.index(answer)
        pred = test_item(inputs, label, gpt3)

        average_meter.update(pred == label)
        logger.info(f"Episode {i}: {choices[pred]} vs {answer}, Avg Acc: {average_meter.avg}")



if __name__ == '__main__':
    import sys
    args = parser.parse_args()

    captioner = MewlCaptioner()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(f"{args.log_path}{args.task_name}.log"))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    gpt3 = openai.Completion
    average_meter = AverageMeter()
    test_task(captioner, args.dataset_path, args.task_name, logger, average_meter, gpt3)