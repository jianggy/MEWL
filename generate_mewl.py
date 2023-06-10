import numpy as np
import json
import random
import itertools
import os
from pprint import pprint
import argparse
import utils
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, required=True,
                    help='type of the episode')
parser.add_argument('--save_path', type=str,
                    default='./rendered', help='path to save')
parser.add_argument('--log_path', type=str,
                    default='./generate.log', help='path to save')
parser.add_argument('--start_idx', type=int, default=0, help='start index')
parser.add_argument('--end_idx', type=int, default=100, help='end index')

DEBUG = False
# DEBUG = True
if not DEBUG:
    from render_images import render_scene
else:
    def render_scene(*args, **kwargs):
        pass
'''
Most common syllabes in English language
adopted from https://github.com/anticoders/fake-words
'''

syllabes = [
    'the', 'ing', 'er', 'a', 'ly', 'ed', 'i', 'es', 're', 'tion', 'in', 'e', 'con', 'y', 'ter', 'ex', 'al', 'de', 'com', 'o', 'di', 'en', 'an', 'ty', 'ry', 'u',
    'ti', 'ri', 'be', 'per', 'to', 'pro', 'ac', 'ad', 'ar', 'ers', 'ment', 'or', 'tions', 'ble', 'der', 'ma', 'na', 'si', 'un', 'at', 'dis', 'ca', 'cal', 'man', 'ap',
    'po', 'sion', 'vi', 'el', 'est', 'la', 'lar', 'pa', 'ture', 'for', 'is', 'mer', 'pe', 'ra', 'so', 'ta', 'as', 'col', 'fi', 'ful', 'get', 'low', 'ni', 'par', 'son',
    'tle', 'day', 'ny', 'pen', 'pre', 'tive', 'car', 'ci', 'mo', 'an', 'aus', 'pi', 'se', 'ten', 'tor', 'ver', 'ber', 'can', 'dy', 'et', 'it', 'mu', 'no', 'ple', 'cu',
    'fac', 'fer', 'gen', 'ic', 'land', 'light', 'ob', 'of', 'pos', 'tain', 'den', 'ings', 'mag', 'ments', 'set', 'some', 'sub', 'sur', 'ters', 'tu', 'af', 'au', 'cy', 'fa', 'im',
    'li', 'lo', 'men', 'min', 'mon', 'op', 'out', 'rec', 'ro', 'sen', 'side', 'tal', 'tic', 'ties', 'ward', 'age', 'ba', 'but', 'cit', 'cle', 'co', 'cov', 'daq', 'dif', 'ence',
    'ern', 'eve', 'hap', 'ies', 'ket', 'lec', 'main', 'mar', 'mis', 'my', 'nal', 'ness', 'ning', 'nu', 'oc', 'pres', 'sup', 'te', 'ted', 'tem', 'tin', 'tri', 'tro', 'up',
]


def random_lexicon(size=48, n_syllabes=3):
    '''create constructed language words, in natural language'''
    nouns = set()

    while len(nouns) != size:
        nouns.add(''.join(np.random.choice(syllabes, n_syllabes)))

    return list(nouns)


def get_attributes(data_path='./data/properties.json'):
    '''read available properties'''
    with open(data_path, 'r') as f:
        data = json.load(f)

    attributes = {attr: list(data[attr].keys()) for attr in data}

    return attributes


class MewlBasic:
    def __init__(self, selected_attribute, context_size=6, lexicon_size=3):
        self.attributes = get_attributes()
        self.selected_attribute = selected_attribute
        all_attrs = self.attributes[selected_attribute]
        self.attrs = random.sample(all_attrs, lexicon_size)
        self.attributes[selected_attribute] = self.attrs
        self.context_size = context_size
        self.lexicon_size = lexicon_size
        self.lexicons, self.objects = self.generate_lexicon_concept()
        self.scenes = list(itertools.combinations(
            self.objects, 1))
        self.info = {}

    def generate_lexicon_concept(self):
        attributes = self.attributes
        # expand all attributes
        unique_attributes = list(itertools.product(*attributes.values()))

        unique_names = random_lexicon(self.lexicon_size, n_syllabes=2)
        lexicon = dict(zip(unique_names, self.attrs))
        attr2name = {v: k for k, v in lexicon.items()}

        unique_objects = []
        attr_idx = list(attributes.keys()).index(self.selected_attribute)

        for i, obj in enumerate(unique_attributes):
            obj_attr = obj[attr_idx]
            if obj_attr not in attr2name:
                continue
            obj_name = f"{attr2name[obj_attr]} #{i}"
            unique_objects.append((obj_name, obj))

        return lexicon, dict(unique_objects)

    def generate_context(self, save_path):

        def no_shared_common():
            groups = {}
            for image in self.scenes[:self.context_size]:
                for obj in image:
                    if self.objects[obj][attr_idx] in groups:
                        groups[self.objects[obj][attr_idx]].append(
                            self.objects[obj])
                    else:
                        groups[self.objects[obj][attr_idx]] = [
                            self.objects[obj]]
            group_shared_indices = []
            for _, group in groups.items():
                shared_indices = []
                for index in range(4):
                    if len(set([obj[index] for obj in group])) == 1:
                        shared_indices.append(index)
                group_shared_indices.append(set(shared_indices))
            return len(set.intersection(*group_shared_indices)) == 1

        self.info['objects'] = self.objects
        self.info['word-concept'] = self.lexicons
        self.info['context_panels'] = []

        random.shuffle(self.scenes)

        attr_idx = list(self.attributes.keys()).index(self.selected_attribute)
        # make sure all attrs are in the context
        while True:
            attrs = set()
            for image in self.scenes[:self.context_size]:
                for obj in image:
                    attrs.add(self.objects[obj][attr_idx])
            if len(attrs) == self.lexicon_size and no_shared_common():
                break
            else:
                random.shuffle(self.scenes)

        # generate contextual images
        for image in self.scenes[:self.context_size]:
            t = [self.objects[obj] for obj in image]
            name = ' and '.join(image)
            scene_struct = render_scene(
                list(zip(image, t)), output_image=f'{save_path}/{name}.png')
            self.info['context_panels'].append({
                'name': name,
                'objects': t,
            })

            with open(f'{save_path}/{name}.json', 'w') as f:
                json.dump(scene_struct, f, indent=4)

    def generate_query(self, save_path):
        # generate query image
        t = [self.objects[obj]
             for obj in self.scenes[self.context_size]]
        ground_truth = self.scenes[self.context_size][0].split(' ')[0]

        choices = set(list(self.lexicons.keys()))
        while len(choices) < 5:
            choices.add(random_lexicon(1, n_syllabes=2)[0])

        choices = list(choices)
        random.shuffle(choices)

        scene_struct = render_scene(
            list(zip(self.scenes[self.context_size], t)), output_image=f'{save_path}/question.png')

        with open(f'{save_path}/question.json', 'w') as f:
            json.dump(scene_struct, f, indent=4)

        self.info['question'] = {
            'objects': t,
            'choices': choices,
            'answer': ground_truth,
        }

    def save_json(self, save_path):
        # write info to json
        with open(f'{save_path}/info.json', 'w') as f:
            json.dump(self.info, f, indent=4)


class MewlObject:
    def __init__(self, context_size=6, lexicon_size=6, co_cnt=3):
        self.attributes = get_attributes()
        self.context_size = context_size
        self.lexicon_size = lexicon_size
        self.co_cnt = co_cnt
        self.lexicons = self.generate_lexicon_concept()
        self.objects = self.lexicons
        self.scenes = list(itertools.combinations(
            self.objects, self.co_cnt))
        self.info = {}

    def generate_lexicon_concept(self):
        attributes = self.attributes
        # expand all attributes
        unique_attributes = list(itertools.product(*attributes.values()))

        unique_names = random_lexicon(len(unique_attributes))
        unique_objects = list(zip(unique_names, unique_attributes))

        lexicons = random.sample(unique_objects, self.lexicon_size)

        return dict(lexicons)

    def generate_context(self, save_path):
        random.shuffle(self.scenes)
        self.info['objects'] = self.objects
        self.info['word-concept'] = {k: list(v)
                                     for k, v in self.lexicons.items()}
        self.info['context_panels'] = []

        # generate contextual images
        for image in self.scenes[:self.context_size]:
            t = [self.objects[obj] for obj in image]
            name = ' and '.join(image)
            scene_struct = render_scene(
                list(zip(image, t)), output_image=f'{save_path}/{name}.png')
            self.info['context_panels'].append({
                'name': name,
                'objects': t,
            })

            with open(f'{save_path}/{name}.json', 'w') as f:
                json.dump(scene_struct, f, indent=4)

    def generate_query(self, save_path):
        # generate query image
        t = [self.objects[obj]
             for obj in self.scenes[self.context_size]]
        ground_truth = ' and '.join(self.scenes[self.context_size])
        choices = [' and '.join(scene)
                   for scene in self.scenes[self.context_size: self.context_size+5]]
        random.shuffle(choices)

        scene_struct = render_scene(
            list(zip(self.scenes[self.context_size], t)), output_image=f'{save_path}/question.png')

        with open(f'{save_path}/question.json', 'w') as f:
            json.dump(scene_struct, f, indent=4)

        self.info['question'] = {
            'objects': t,
            'choices': choices,
            'answer': ground_truth,
        }

    def save_json(self, save_path):
        # write info to json
        with open(f'{save_path}/info.json', 'w') as f:
            json.dump(self.info, f, indent=4)


class MewlComposite:
    def __init__(self, context_size=6, lexicon_size=6, attr_cnt=2):
        self.attributes = get_attributes()
        self.context_size = context_size
        self.attr_cnt = attr_cnt
        self.lexicon_size = lexicon_size
        self.lexicons, self.alter_attr_type, self.alter_attr_values = self.generate_lexicon_concept()
        self.info = {}

    def generate_lexicon_concept(self):
        attributes = self.attributes
        attributes_except_size = list(attributes.keys())
        attributes_except_size.remove('sizes')

        # sample n_attr attributes
        alter_attr_type = random.sample(attributes_except_size, self.attr_cnt)

        # sample n_attr values for each attribute
        assert self.lexicon_size % self.attr_cnt == 0
        alter_attr_values = [random.sample(
            attributes[attr], (self.lexicon_size // self.attr_cnt)) for attr in alter_attr_type]

        # bind name to alter_attr_values
        alter_attr_names = random_lexicon(
            sum(len(i) for i in alter_attr_values))
        lexicons = dict(zip(alter_attr_names, list(
            attr for attr_type in alter_attr_values for attr in attr_type)))

        return lexicons, alter_attr_type, alter_attr_values

    def generate_context(self, save_path):

        def no_shared_common():
            for alter_type in self.alter_attr_type:
                groups = {}
                for name, obj_attr in self.scenes[:self.context_size]:
                    group_key = obj_attr[0][list(attributes.keys()).index(alter_type)]
                    if group_key in groups:
                        groups[group_key].append(obj_attr[0])
                    else:
                        groups[group_key] = [obj_attr[0]]
                group_shared_indices = []
                for _, group in groups.items():
                    shared_indices = []
                    for index in range(4):
                        if len(set([obj[index] for obj in group])) == 1:
                            shared_indices.append(index)
                    group_shared_indices.append(set(shared_indices))
                if len(set.intersection(*group_shared_indices)) != 1:
                    return False
            return True

        attributes = self.attributes
        self.info['word-concept'] = self.lexicons
        self.info['context_panels'] = []

        reversed_dict = {v: k for k, v in self.lexicons.items()}

        # generate unique objects
        while True:
            self.scenes = []
            for obj in itertools.product(*self.alter_attr_values):
                name = []
                obj_attr = [None] * len(attributes.keys())
                for idx, attr in enumerate(obj):
                    obj_attr[list(attributes.keys()).index(
                        self.alter_attr_type[idx])] = attr
                    name.append(reversed_dict[attr])
                # random sample other attributes
                for idx, attr in enumerate(obj_attr):
                    if attr is None:
                        obj_attr[idx] = random.choice(
                            list(attributes.values())[idx])

                self.scenes.append((' '.join(name), [obj_attr]))
            if no_shared_common():
                break

        random.shuffle(self.scenes)

        self.info['objects'] = {k: v[0] for k, v in self.scenes}
        # generate contextual images
        for name, t in self.scenes[:self.context_size]:
            scene_struct = render_scene(
                list(zip([name], t)), output_image=f'{save_path}/{name}.png')
            self.info['context_panels'].append({
                'name': name,
                'objects': t,
            })

            with open(f'{save_path}/{name}.json', 'w') as f:
                json.dump(scene_struct, f, indent=4)

    def generate_query(self, save_path):
        assert self.scenes
        # generate query image
        t = self.scenes[self.context_size][1]
        ground_truth = self.scenes[self.context_size][0]
        choices = set(scene[0] for scene in self.scenes[self.context_size:])
        while len(choices) < 5:
            choices.add(self.scenes[random.randint(0, len(self.scenes)-1)][0])

        choices = list(choices)
        random.shuffle(choices)

        scene_struct = render_scene(
            list(zip([ground_truth], t)), output_image=f'{save_path}/question.png')

        with open(f'{save_path}/question.json', 'w') as f:
            json.dump(scene_struct, f, indent=4)

        self.info['question'] = {
            'objects': t,
            'choices': choices,
            'answer': ground_truth,
        }

    def save_json(self, save_path):
        # write info to json
        with open(f'{save_path}/info.json', 'w') as f:
            json.dump(self.info, f, indent=4)


class MewlBootstrap:
    def __init__(self, context_size=6, lexicon_size=6, co_cnt=2, distractor_cnt=1):
        self.attributes = get_attributes()
        self.context_size = context_size
        self.lexicon_size = lexicon_size
        self.co_cnt = co_cnt
        self.distractor_cnt = distractor_cnt
        self.lexicons = self.generate_lexicon_concept()
        self.objects = self.lexicons
        self.scenes = list(itertools.combinations(
            self.objects, self.co_cnt))
        self.info = {}

    def generate_lexicon_concept(self):
        attributes = self.attributes
        # expand all attributes
        unique_attributes = list(itertools.product(*attributes.values()))

        unique_names = random_lexicon(len(unique_attributes))
        unique_objects = list(zip(unique_names, unique_attributes))

        lexicons = random.sample(unique_objects, self.lexicon_size)

        return dict(lexicons)

    def generate_context(self, save_path):
        random.shuffle(self.scenes)

        self.info['objects'] = self.objects
        self.info['word-concept'] = {k: list(v)
                                        for k, v in self.lexicons.items()}
        self.info['context_panels'] = []

        # generate contextual images

        objects_for_context_scenes = set()
        for image in self.scenes[:self.context_size]:
            objects_for_context_scenes.update([self.objects[obj] for obj in image])
        all_distractors = set(itertools.product(*self.attributes.values())).difference(objects_for_context_scenes)

        for image in self.scenes[:self.context_size]:
            t = [self.objects[obj] for obj in image]
            # add distractors with self.attributes
            distractors = random.sample(
                list(all_distractors), self.distractor_cnt)
            t += distractors

            name = ' and '.join(image)
            scene_struct = render_scene(
                list(zip(image + (None,) * self.distractor_cnt, t)), output_image=f'{save_path}/{name}.png')

            # get relations from scene_struct
            relative_rel = None
            all_relations = list(scene_struct['relationships'])
            random.shuffle(all_relations)

            for relation in all_relations:
                if image[1] in scene_struct['relationships'][relation][image[0]]:
                    relative_rel = relation
                    break

            new_name = f"{image[1]} {relative_rel} {image[0]}"

            # rename image
            shutil.move(f'{save_path}/{name}.png',
                        f'{save_path}/{new_name}.png')

            self.info['context_panels'].append({
                'name': new_name,
                'objects': t,
            })

            with open(f'{save_path}/{new_name}.json', 'w') as f:
                json.dump(scene_struct, f, indent=4)

    def generate_query(self, save_path):
        # generate query image
        t = [self.objects[obj]
             for obj in self.scenes[self.context_size]]
            
        objects_for_context_scenes = set()
        for image in self.scenes[:self.context_size]:
            objects_for_context_scenes.update([self.objects[obj] for obj in image])
        all_distractors = set(itertools.product(*self.attributes.values())).difference(objects_for_context_scenes)

        distractors = random.sample(
            list(all_distractors), self.distractor_cnt)
        t += distractors

        scene_struct = render_scene(
            list(zip(self.scenes[self.context_size] + (None, ) * self.distractor_cnt, t)), output_image=f'{save_path}/question.png')

        # get relations from scene_struct
        relative_rel = None
        all_relations = list(scene_struct['relationships'])
        random.shuffle(all_relations)

        for relation in all_relations:
            if self.scenes[self.context_size][1] in scene_struct['relationships'][relation][self.scenes[self.context_size][0]]:
                relative_rel = relation
                break

        ground_truth = f"{self.scenes[self.context_size][1]} {relative_rel} {self.scenes[self.context_size][0]}"

        choices = []
        for scene in self.scenes[self.context_size+1: self.context_size+5]:
            while True:
                random_direction = random.choice(all_relations)
                candidate = f' {random_direction} '.join(scene)
                if candidate != ground_truth:
                    choices.append(candidate)
                    break

        choices.append(ground_truth)
        random.shuffle(choices)

        with open(f'{save_path}/question.json', 'w') as f:
            json.dump(scene_struct, f, indent=4)

        self.info['question'] = {
            'objects': t,
            'choices': choices,
            'answer': ground_truth,
        }

    def save_json(self, save_path):
        # write info to json
        with open(f'{save_path}/info.json', 'w') as f:
            json.dump(self.info, f, indent=4)


class MewlRelation:
    def __init__(self, context_size=6, lexicon_size=3, co_cnt=3, relations=["behind", "front", "left", "right"]):
        self.attributes = get_attributes()
        self.context_size = context_size
        self.co_cnt = co_cnt
        self.relations = random.sample(relations, lexicon_size)
        self.lexicon_size = lexicon_size
        self.lexicons = self.generate_lexicon_concept()
        self.named_attributes = ["colors", "shapes"]
        self.info = {}

        # named attributes
        self.unique_named_objects = list(itertools.product(
            *list(self.attributes[attr] for attr in self.named_attributes)))

    def generate_lexicon_concept(self):
        attributes = self.attributes

        unique_names = random_lexicon(len(self.relations))
        lexicons = list(zip(unique_names, self.relations))

        return dict(lexicons)

    def generate_context(self, save_path):
        equiv_dict = {"behind": "front", "front": "behind",
                      "left": "right", "right": "left"}
        self.info['objects'] = {}
        self.info['word-concept'] = {k: v
                                     for k, v in self.lexicons.items()}
        reversed_dict = {v: k for k, v in self.lexicons.items()}

        self.info['context_panels'] = []
        all_names = set()

        last_relation = None
        # generate contextual images
        while len(all_names) < self.context_size:
            context_idx = len(all_names)
            # sample named objects
            sampled_t_named_attributes = random.sample(
                self.unique_named_objects, self.co_cnt)
            t = []
            # obj_name, (shape_name, color_name, mat_name, size_name)
            for color, shape in sampled_t_named_attributes:
                t.append((shape, color, random.choice(
                    self.attributes['materials']), random.choice(self.attributes['sizes'])))

            image = [f"{obj[1]} {obj[0]}" for obj in t]

            name = ' and '.join(image)

            scene_struct = render_scene(
                list(zip(image, t)), output_image=f'{save_path}/{name}.png')

            # get relations from scene_struct
            target_relation = self.relations[int(context_idx / 2)]
            all_relations = list(scene_struct['relationships'])

            name_0 = None
            name_1 = None
            all_relation_items = list(scene_struct['relationships'][target_relation].items())

            dead_cnt = 0
            while True:
                random.shuffle(all_relation_items)
                for obj_0, obj_1s in all_relation_items:
                    if len(obj_1s) > 0:
                        name_0 = obj_0
                        name_1 = random.choice(obj_1s)
                        break

                for relation in all_relations:
                    if name_1 in scene_struct['relationships'][relation][name_0] and relation != target_relation:
                        other_relation = relation
                        break

                assert other_relation is not None

                if context_idx % 2 == 1 and last_relation == other_relation:
                    print("Same relation, regenerate!!!!!")
                    dead_cnt += 1
                    if dead_cnt > 5:
                        print("Dead loop!!!!!")
                        os.remove(f'{save_path}/{name}.png')
                        break
                else:
                    last_relation = other_relation
                    break
                
            assert name_0 is not None and name_1 is not None


            if dead_cnt > 5:
                continue

            new_name = f"{name_1} {reversed_dict[target_relation]} {name_0}"
            if new_name in all_names:
                os.remove(f'{save_path}/{name}.png')
                continue
            else:
                all_names.add(new_name)

            for i_name, obj in zip(image, t):
                self.info['objects'][i_name] = obj

            # rename image
            shutil.move(f'{save_path}/{name}.png',
                        f'{save_path}/{new_name}.png')

            self.info['context_panels'].append({
                'name': new_name,
                'objects': t,
            })

            with open(f'{save_path}/{new_name}.json', 'w') as f:
                json.dump(scene_struct, f, indent=4)

    def generate_query(self, save_path):
        reversed_dict = {v: k for k, v in self.lexicons.items()}

        # sample named objects
        sampled_t_named_attributes = random.sample(
            self.unique_named_objects, self.co_cnt)
        t = []
        # obj_name, (shape_name, color_name, mat_name, size_name)
        for color, shape in sampled_t_named_attributes:
            t.append((shape, color, random.choice(
                self.attributes['materials']), random.choice(self.attributes['sizes'])))

        image = [f"{obj[1]} {obj[0]}" for obj in t]

        for name, obj in zip(image, t):
            self.info['objects'][name] = obj

        scene_struct = render_scene(
            list(zip(image, t)), output_image=f'{save_path}/question.png')

        # get relations from scene_struct
        name_0 = None
        name_1 = None
        all_ground_truth = set()

        for relation in self.relations:
            relation_items = list(scene_struct['relationships'][relation].items())
            random.shuffle(relation_items)
            for obj_0, obj_1s in relation_items:
                if len(obj_1s) > 0:
                    name_0 = obj_0
                    for name_1 in obj_1s:
                        all_ground_truth.add(f"{name_1} {reversed_dict[relation]} {name_0}")


        ground_truth = random.choice(list(all_ground_truth))

        choices = set()
        choices.add(ground_truth)

        while (len(choices) < 5):
            candidate = image.copy()
            random.shuffle(candidate)
            random_description = f"{candidate[1]} {random.choice(list(reversed_dict.values()))} {candidate[0]}"
            if random_description not in all_ground_truth:
                choices.add(random_description)

        choices = list(choices)
        random.shuffle(choices)

        with open(f'{save_path}/question.json', 'w') as f:
            json.dump(scene_struct, f, indent=4)

        self.info['question'] = {
            'objects': t,
            'choices': choices,
            'answer': ground_truth,
        }

    def save_json(self, save_path):
        # write info to json
        with open(f'{save_path}/info.json', 'w') as f:
            json.dump(self.info, f, indent=4)


class MewlPragmatic:
    def __init__(self, context_size=6):
        self.attributes = get_attributes()
        self.lexicon_size = context_size
        self.context_size = context_size
        self.lexicons = self.generate_lexicon_concept()
        self.co_cnt = 3
        self.info = {}
        self.scenes = []

    def generate_lexicon_concept(self):
        all_attributes = [(attr_type, attr) for attr_type,
                          attr_list in self.attributes.items() for attr in attr_list]
        selected_attributes = random.sample(all_attributes, self.lexicon_size)
        unique_names = random_lexicon(self.lexicon_size, n_syllabes=2)
        lexicons = list(zip(unique_names, selected_attributes))

        return dict(lexicons)

    def generate_context(self, save_path):
        self.info['objects'] = {}
        self.info['word-concept'] = {k: v[1] for k, v in self.lexicons.items()}
        all_objects = list(itertools.product(*self.attributes.values()))

        reversed_dict = {v: k for k, v in self.lexicons.items()}

        self.info['context_panels'] = []
        for panel_idx, alter_att in enumerate(self.lexicons.items()):
            names = [None, None, None, None]
            tgt_name, (alter_att_type, alter_att_name) = alter_att
            alter_att_idx = list(self.attributes.keys()).index(alter_att_type)

            while True:
                base_obj = random.choice(all_objects)
                tgt_obj = list(base_obj)
                tgt_obj[alter_att_idx] = alter_att_name
                bg_obj = list(base_obj)
                bg_alter_att_idx = random.choice(
                    [i for i in range(len(self.attributes)) if i != alter_att_idx])
                bg_obj[bg_alter_att_idx] = random.choice(
                    list(self.attributes.values())[bg_alter_att_idx])
                tgt_obj, bg_obj = tuple(tgt_obj), tuple(bg_obj)

                if base_obj != tgt_obj and base_obj != bg_obj and tgt_obj != bg_obj:
                    break

            objects = [base_obj, tgt_obj, bg_obj]
            random.shuffle(objects)

            arrow_idx = objects.index(tgt_obj)
            names[arrow_idx] = tgt_name
            objects.append(('arrow', 'yellow', 'rubber', 'small'))
            self.scenes.append((names, objects, arrow_idx))

            self.info['objects'][tgt_name] = tgt_obj

        random.shuffle(self.scenes)

        for names, objects, arrow_idx in self.scenes:
            scene_struct = render_scene(list(zip(
                names, objects)), output_image=f'{save_path}/{names[arrow_idx]}.png', arrow_idx=arrow_idx)
            self.info['context_panels'].append({
                'name': names[arrow_idx],
                'objects': objects,
                'arrow_idx': arrow_idx,
            })

            with open(f'{save_path}/{names[arrow_idx]}.json', 'w') as f:
                json.dump(scene_struct, f, indent=4)

    def generate_query(self, save_path):
        all_objects = list(itertools.product(*self.attributes.values()))
        alter_att = random.choice(list(self.lexicons.items()))

        names = [None, None, None, None]
        tgt_name, (alter_att_type, alter_att_name) = alter_att
        alter_att_idx = list(self.attributes.keys()).index(alter_att_type)

        while True:
            base_obj = random.choice(all_objects)
            tgt_obj = list(base_obj)
            tgt_obj[alter_att_idx] = alter_att_name
            bg_obj = list(base_obj)
            bg_alter_att_idx = random.choice(
                [i for i in range(len(self.attributes)) if i != alter_att_idx])
            bg_obj[bg_alter_att_idx] = random.choice(
                list(self.attributes.values())[bg_alter_att_idx])
            tgt_obj, bg_obj = tuple(tgt_obj), tuple(bg_obj)

            if base_obj != tgt_obj and base_obj != bg_obj and tgt_obj != bg_obj:
                break

        objects = [base_obj, tgt_obj, bg_obj]
        random.shuffle(objects)

        arrow_idx = objects.index(tgt_obj)
        names[arrow_idx] = tgt_name
        objects.append(('arrow', 'yellow', 'rubber', 'small'))
        self.scenes.append((tgt_name, objects, arrow_idx))

        self.info['objects'][tgt_name] = tgt_obj

        scene_struct = render_scene(list(zip(
            names, objects)), output_image=f'{save_path}/question.png', arrow_idx=arrow_idx)

        with open(f'{save_path}/question.json', 'w') as f:
            json.dump(scene_struct, f, indent=4)

        ground_truth = tgt_name
        choices = set([tgt_name])

        while len(choices) < 5:
            choices.add(random.choice(list(self.lexicons.keys())))

        choices = list(choices)
        random.shuffle(choices)

        self.info['question'] = {
            'objects': objects,
            'choices': choices,
            'answer': ground_truth,
            'arrow_idx': arrow_idx,
        }

    def save_json(self, save_path):
        with open(f'{save_path}/info.json', 'w') as f:
            json.dump(self.info, f, indent=4)


class MewlNumber:
    def __init__(self, context_size=6):
        self.attributes = get_attributes()
        self.lexicon_size = context_size
        self.context_size = context_size
        self.lexicons, self.objects = self.generate_lexicon_concept()
        self.scenes = []
        self.info = {}

    def generate_lexicon_concept(self):
        unique_names = random_lexicon(self.lexicon_size, n_syllabes=2)
        unique_objects = list(itertools.product(*self.attributes.values()))

        lexicons = zip(unique_names, range(1, self.lexicon_size+1))

        return dict(lexicons), unique_objects

    def generate_context(self, save_path):
        self.info['objects'] = {}
        self.info['word-concept'] = self.lexicons
        self.info['context_panels'] = []

        for name, number in self.lexicons.items():
            t = random.sample(self.objects, number)
            for obj in t:
                self.info['objects'][f"#{len(self.info['objects'])}"] = obj
            self.scenes.append((name, t))

        random.shuffle(self.scenes)
        for name, objects in self.scenes:
            scene_struct = render_scene(list(
                zip([None, ]*len(objects), objects)), output_image=f'{save_path}/{name}.png')
            self.info['context_panels'].append({
                'name': name,
                'objects': objects,
            })

            with open(f'{save_path}/{name}.json', 'w') as f:
                json.dump(scene_struct, f, indent=4)

    def generate_query(self, save_path):
        name, number = random.choice(list(self.lexicons.items()))
        t = random.sample(self.objects, number)
        for obj in t:
            self.info['objects'][f"#{len(self.info['objects'])}"] = obj
        self.scenes.append((name, t))

        scene_struct = render_scene(
            list(zip([None, ]*len(t), t)), output_image=f'{save_path}/question.png')

        with open(f'{save_path}/question.json', 'w') as f:
            json.dump(scene_struct, f, indent=4)

        ground_truth = name
        choices = set([name])

        while len(choices) < 5:
            choices.add(random.choice(list(self.lexicons.keys())))

        choices = list(choices)
        random.shuffle(choices)

        self.info['question'] = {
            'objects': t,
            'choices': choices,
            'answer': ground_truth,
        }

    def save_json(self, save_path):
        with open(f'{save_path}/info.json', 'w') as f:
            json.dump(self.info, f, indent=4)


if __name__ == '__main__':
    import logging
    import time
    import sys
    
    argv = utils.extract_args()
    args = parser.parse_args(argv)

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=args.log_path,
                        level=logging.DEBUG, format=LOG_FORMAT)

    if args.type in ['composite', 'object', 'bootstrap', 'relation', 'pragmatic', 'number']:
        logging.info(f"Generating {args.type}")
        for i in range(args.start_idx, args.end_idx):
            start = time.time()
            save_path = os.path.join(args.save_path, f'{args.type}/{i}')
            if os.path.exists(save_path):
                shutil.rmtree(save_path)

            os.makedirs(save_path, exist_ok=True)
            episode = getattr(sys.modules[__name__], 'Mewl' + args.type.capitalize())()
            episode.generate_context(save_path)
            episode.generate_query(save_path)
            episode.save_json(save_path)
            logging.info(f"Done {i} in {time.time() - start}")
    elif args.type in ['color', 'material', 'shape']:
        logging.info(f"Generating {args.type}")
        for i in range(args.start_idx, args.end_idx):
            start = time.time()
            save_path = os.path.join(args.save_path, f'{args.type}/{i}')
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            
            os.makedirs(save_path, exist_ok=True)
            episode = MewlBasic(args.type + 's')
            episode.generate_context(save_path)
            episode.generate_query(save_path)
            episode.save_json(save_path)
            logging.info(f"Done {i} in {time.time() - start}")
    else:
        raise NotImplementedError