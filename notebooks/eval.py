from src.utils import RAW_DATA_PATH, read_pt, write_pt, read_json
import torch
import re


RICO_LABEL2ID = {
    'text button': 0,
    'background image': 1,
    'icon': 2,
    'list item': 3,
    'text': 4,
    'toolbar': 5,
    'web view': 6,
    'input': 7,
    'card': 8,
    'advertisement': 9,
    'image': 10,
    'drawer': 11,
    'radio button': 12,
    'checkbox': 13,
    'multi-tab': 14,
    'pager indicator': 15,
    'modal': 16,
    'on/off switch': 17,
    'slider': 18,
    'map view': 19,
    'button bar': 20,
    'video': 21,
    'bottom navigation': 22,
    'number stepper': 23,
    'date picker': 24,
}


WEB_LABEL2ID = {
    'text': 0,
    'link': 1,
    'button': 2,
    'title': 3,
    'description': 4,
    'image': 5,
    'background': 6,
    'logo': 7,
    'icon': 8,
    'input': 9,
    'select': 10,
    'textarea': 11
}


LABEL2ID = {
    'web': WEB_LABEL2ID,
    'rico': RICO_LABEL2ID
}


LABEL = {
    'web': list(WEB_LABEL2ID.keys()),
    'rico': list(RICO_LABEL2ID.keys())
}


CANVAS_SIZE = {
    'web': (120, 120),
    'rico': (144, 256)
}

def _collect_attribute(seq_list, dataset):
    label, pos = [], []
    for layout_seq in seq_list:
        _label, _pos = _findall_elements(layout_seq, dataset)
        label.append(_label)
        pos.append(_pos)
    pos_t = torch.zeros(len(label), 200, 4)
    label_t = torch.zeros(len(label), 200).long()
    padding_mask_t = torch.zeros(len(label), 200).bool()
    for i, (_label, _pos) in enumerate(zip(label, pos)):
        if len(_label) == 0: continue
        pos_t[i][0: len(_label)] = convert_ltwh_to_ltrb(_pos)
        label_t[i][0: len(_label)] = _label
        padding_mask_t[i][0: len(_label)] = 1
    return pos, label, pos_t, label_t, padding_mask_t


def convert_ltwh_to_ltrb(bbox):
    l, t, w, h = decapulate(bbox)
    r = l + w
    b = t + h
    return torch.stack([l, t, r, b], axis=-1)

def decapulate(bbox):
    if len(bbox.size()) == 2:
        x1, y1, x2, y2 = bbox.T
    else:
        x1, y1, x2, y2 = bbox.permute(2, 0, 1)
    return x1, y1, x2, y2

def _findall_elements(s, dataset):
    labels = LABEL[dataset]
    canvas_width, canvas_height = CANVAS_SIZE[dataset]
    element_pattern = '(' + '|'.join(labels) + ')' + r' (\d+) (\d+) (\d+) (\d+)'
    label2id = LABEL2ID[dataset]

    elements = re.findall(element_pattern, s)
    label = torch.tensor(
        [label2id[element[0]] for element in elements]
    )
    position = torch.tensor([
        [int(element[1]) / canvas_width,
        int(element[2]) / canvas_height,
        int(element[3]) / canvas_width,
        int(element[4]) / canvas_height]
        for element in elements
    ])
    return label, position



training_golds = read_json('/scratch/muyanchen/LayoutGeneration-main/LayoutPrompter/dataset/webui/raw/train.json')
gold_layout_seq = [d['plain_layout_seq'] for d in training_golds]

gold_pos, gold_label, gold_pos_t, gold_label_t, gold_padding_mask = _collect_attribute(gold_layout_seq, 'web')
print(1)