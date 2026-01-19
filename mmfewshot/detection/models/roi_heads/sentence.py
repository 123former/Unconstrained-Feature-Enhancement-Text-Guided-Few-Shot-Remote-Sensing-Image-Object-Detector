import pdb
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
from torch.nn.utils.rnn import pad_sequence
import pdb
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sentences = {
    "airplane": "An airplane includes fuselage, wings, tail, and engines, etc. The two wings of the aircraft are symmetrically distributed on both sides of the fuselage.",
    "baseballfield": "A baseball diamond is a quarter circle and usually comes in pairs. It is covered by grass.",
    "basketballcourt": "The basketball court is usually close to ground track fields and consists of a rectangular floor with baskets at each end. Outdoor surfaces are generally made from standard paving materials.",
    "bridge": "A bridge is usually slender. It is a structure built to span a physical obstacle such as a body of water, valley, road, or rail, without blocking the way underneath.",
    "groundtrackfield": "A ground track field contains the outer oval shaped running track and an area of turf within this track.",
    "harbor": "A harbor is a sheltered body of water. It provides docking areas for ships, boats, and barges.",
    "ship": "Ships are usually found in oceans, rivers, lakes, etc. where there is water. The shape of a ship is a long bar with pointed ends.",
    "storagetank": "Storage tanks are usually vertical cylindrical. They are white or gray. Multiple tanks usually appear together.",
    "tenniscourt": "A tennis court is the venue where the sport of tennis is played. It is a firm rectangular surface with a low net stretched across the centre.",
    "vehicle": "The vehicle contains four tires and has windshields front and rear. Vehicles drive on the road or are stationed in the parking lot.",
    "airport": "An airport consists of a landing area with airport aprons and runways for planes to take off and to land. It includes adjacent utility buildings such as control towers, hangars and terminals.",
    "chimney": "Chimneys made of masonry, clay or metal are usually conical in shape, large below and small above. They are typically vertical, or as close to vertical as possible.",
    "dam": "Dams are usually built along the water edge, narrow at the top and wide at the bottom. They are barriers that prevent or restrict the flow of surface water or underground streams.",
    "Expressway-Service-area": "An expressway service area is usually distributed on both sides of the road. It includes parking loes and buildings for people to rest and shopping.",
    "Expressway-toll-station": "An expressway toll station is an enclosure placed along a toll road and consists of several adjacent tell booths and spans the entire highway.",
    "golffield": "A golf field is halfway up the mountain and usually covers a wide area with many fairways intertwined.",
    "overpass": "An overpass is a bridge or road that crosses over another road or railway. An overpass and underpass together form a grade separation.",
    "stadium": "Stadium are usually open air sports venues. It is round or oval. It contains auditoriums, a running track, and a football pitch.",
    "trainstation": "A train station generally consists of multiple platforms, multiple tracks and a station building. Trains stop at the train station.",
    "windmill": "A windmill is a structure that contains three blades which are erected by a tall column."
}

VHR10_SPLIT = dict(
    ALL_CLASSES_SPLIT1=('basketballcourt', 'bridge', 'groundtrackfield',
                        'harbor', 'ship', 'storagetank', 'vehicle', 'airplane', 'baseballfield', 'tenniscourt'),
    NOVEL_CLASSES_SPLIT1=('airplane', 'baseballfield', 'tenniscourt'),
    BASE_CLASSES_SPLIT1=('basketballcourt', 'bridge', 'groundtrackfield',
                         'harbor', 'ship', 'storagetank', 'vehicle'),

    ALL_CLASSES_SPLIT2=('airplane', 'baseballfield', 'bridge',
                        'harbor', 'ship', 'storagetank', 'tenniscourt', 'basketballcourt', 'groundtrackfield',
                        'vehicle'),
    NOVEL_CLASSES_SPLIT2=('basketballcourt', 'groundtrackfield', 'vehicle'),
    BASE_CLASSES_SPLIT2=('airplane', 'baseballfield', 'bridge',
                         'harbor', 'ship', 'storagetank', 'tenniscourt'),

    ALL_CLASSES_SPLIT3=('airplane', 'bridge', 'groundtrackfield',
                         'harbor', 'ship', 'storagetank', 'tenniscourt', 'basketballcourt', 'baseballfield', 'vehicle'),
    NOVEL_CLASSES_SPLIT3=('basketballcourt', 'baseballfield', 'vehicle'),
    BASE_CLASSES_SPLIT3=('airplane', 'bridge', 'groundtrackfield',
                         'harbor', 'ship', 'storagetank', 'tenniscourt'),

)

DIOR_SPLIT = dict(
    ALL_CLASSES_SPLIT1=('airplane', 'airport', 'dam',
                        'Expressway-Service-area', 'Expressway-toll-station',
                        'golffield', 'groundtrackfield', 'harbor', 'overpass',
                        'stadium', 'storagetank', 'tenniscourt',
                        'trainstation', 'vehicle', 'windmill', 'baseballfield',
                        'basketballcourt', 'bridge', 'chimney', 'ship'),
    ALL_CLASSES_SPLIT2=('basketballcourt', 'bridge', 'chimney', 'dam',
                        'Expressway-Service-area', 'golffield', 'overpass',
                        'ship', 'stadium', 'storagetank', 'vehicle', 'baseballfield',
                        'tenniscourt', 'trainstation', 'windmill', 'airplane', 'airport',
                        'Expressway-toll-station', 'harbor', 'groundtrackfield'),
    ALL_CLASSES_SPLIT3=('airplane', 'airport', 'baseballfield',
                        'basketballcourt', 'bridge', 'chimney',
                        'Expressway-Service-area', 'Expressway-toll-station',
                        'groundtrackfield', 'harbor', 'overpass', 'ship',
                        'stadium', 'trainstation', 'windmill', 'dam',
                        'golffield', 'storagetank', 'tenniscourt', 'vehicle'),
    ALL_CLASSES_SPLIT4=('airport', 'basketballcourt', 'bridge', 'chimney',
                        'dam', 'Expressway-toll-station', 'golffield',
                        'groundtrackfield', 'harbor', 'ship', 'storagetank',
                        'airplane', 'baseballfield', 'tenniscourt', 'vehicle',
                        'Expressway-Service-area', 'overpass', 'stadium',
                        'trainstation', 'windmill'),
    ALL_CLASSES_SPLIT5=('airport', 'basketballcourt', 'bridge', 'chimney',
                        'dam', 'Expressway-Service-area',
                        'Expressway-toll-station', 'golffield',
                        'groundtrackfield', 'harbor', 'overpass', 'ship',
                        'stadium', 'storagetank', 'vehicle', 'airplane',
                        'baseballfield', 'tenniscourt', 'trainstation',
                        'windmill'),
    NOVEL_CLASSES_SPLIT1=('baseballfield', 'basketballcourt', 'bridge',
                          'chimney', 'ship'),
    NOVEL_CLASSES_SPLIT2=('airplane', 'airport', 'Expressway-toll-station',
                          'harbor', 'groundtrackfield'),
    NOVEL_CLASSES_SPLIT3=('dam', 'golffield', 'storagetank', 'tenniscourt',
                          'vehicle'),
    NOVEL_CLASSES_SPLIT4=('Expressway-Service-area', 'overpass', 'stadium',
                          'trainstation', 'windmill'),
    NOVEL_CLASSES_SPLIT5=('airplane', 'baseballfield', 'tenniscourt',
                          'trainstation', 'windmill'),
    BASE_CLASSES_SPLIT1=('airplane', 'airport', 'dam',
                         'Expressway-Service-area', 'Expressway-toll-station',
                         'golffield', 'groundtrackfield', 'harbor', 'overpass',
                         'stadium', 'storagetank', 'tenniscourt',
                         'trainstation', 'vehicle', 'windmill'),
    BASE_CLASSES_SPLIT2=('basketballcourt', 'bridge', 'chimney', 'dam',
                        'Expressway-Service-area', 'golffield', 'overpass',
                        'ship', 'stadium', 'storagetank', 'vehicle', 'baseballfield',
                        'tenniscourt', 'trainstation', 'windmill'),
    BASE_CLASSES_SPLIT3=('airplane', 'airport', 'baseballfield',
                         'basketballcourt', 'bridge', 'chimney',
                         'Expressway-Service-area', 'Expressway-toll-station',
                         'groundtrackfield', 'harbor', 'overpass', 'ship',
                         'stadium', 'trainstation', 'windmill'),
    BASE_CLASSES_SPLIT4=('airport', 'basketballcourt', 'bridge', 'chimney',
                         'dam', 'Expressway-toll-station', 'golffield',
                         'groundtrackfield', 'harbor', 'ship', 'storagetank',
                         'airplane', 'baseballfield', 'tenniscourt',
                         'vehicle'),
    BASE_CLASSES_SPLIT5=('airport', 'basketballcourt', 'bridge', 'chimney',
                         'dam', 'Expressway-Service-area',
                         'Expressway-toll-station', 'golffield',
                         'groundtrackfield', 'harbor', 'overpass', 'ship',
                         'stadium', 'storagetank', 'vehicle'))


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class Text_Embedding(nn.Module):
    def __init__(self, output_size=2048, use_dior=True, SPLIT='BASE_CLASSES_SPLIT5', mask_enable=False,
                 constrain=False):
        super(Text_Embedding, self).__init__()
        if use_dior:
            dataset_name = DIOR_SPLIT
        else:
            dataset_name = VHR10_SPLIT
        self.classes = dataset_name[SPLIT]
        self.sentences = []
        self.get_sentences()
        self.max_len = self.get_max_len()
        self.word_size = 768
        # self.proj = nn.Conv2d(channels, 2048, 1, padding=0, bias=False)
        # self.proj = nn.Linear(41 * 768, 2048, bias=False)
        self.proj_fusion = nn.Linear(2048 * 2, 2048, bias=False)
        self.classifier = pipeline(model="/home/f523/disk1/sxp/transformor/bert-base-uncased",
                                   task="feature-extraction", device=0)
        self.tokenizer = AutoTokenizer.from_pretrained("/home/f523/disk1/sxp/transformor/bert-base-uncased")

        self.gru = GRUModel(input_size=self.word_size, hidden_size=64, num_layers=2, output_size=output_size)
        self.word2embd = nn.Embedding(29000, self.word_size)
        self.lstm = nn.LSTM(self.word_size, output_size)
        self.mask_enable = mask_enable
        self.constrain = constrain
        self.lambd = 0.01
        self.bn = nn.BatchNorm1d(output_size, affine=False)
        self.constrain_loss = dict()

    def get_max_len(self):
        max_len = 0
        for key in sentences.keys():
            if len(sentences[key]) > max_len:
                max_len = len(sentences[key])
        return max_len

    def get_sentences(self):
        for cls in self.classes:
            self.sentences.append(sentences[cls])

    def get_sentence_batches(self, gt_labels):
        sen_batches = []

        for label in gt_labels:
            sen_batches.append(self.sentences[label[0].item()])
        return sen_batches

    # def forward(self, input, support_feat, fusion=False):
    #     with torch.no_grad():
    #         encoded_inputs = self.tokenizer(input, max_length=37, padding='max_length', truncation=True,
    #                                         return_tensors="pt", add_special_tokens=False)
    #         pad_sentences = self.tokenizer.batch_decode(encoded_inputs["input_ids"])
    #         encoded_pad = self.tokenizer(pad_sentences, padding=True, truncation=True, return_tensors="pt")
    #         embeddings = self.classifier(pad_sentences, return_tensors=True)
    #         embeddings = torch.cat(embeddings, dim=0)
    #
    #         mask = encoded_pad["attention_mask"]
    #         mask[:, 1:-1] = encoded_inputs["attention_mask"]
    #         embeddings_mask = mask.unsqueeze(2) * embeddings
    #     embeddings_gru = self.gru(embeddings_mask.cuda())
    #     if fusion:
    #         out = self.proj_fusion(torch.cat([embeddings_gru, support_feat], dim=1))
    #     else:
    #         out = embeddings_gru
    #     return out

    @staticmethod
    def reverse_variable(var):
        idx = [i for i in range(var.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx))

        if True:
            idx = idx.cuda()

        inverted_var = var.index_select(0, idx)
        return inverted_var

    # def forward(self, input, support_feat, fusion=False):
    #     encoded_inputs = self.tokenizer(input, max_length=39, padding='max_length', truncation=True,
    #                                     return_tensors="pt")
    #     sentences = encoded_inputs["input_ids"].cuda()
    #     sentences = sentences.transpose(0, 1)
    #     word_embeddings = F.tanh(self.word2embd(sentences))
    #     if self.mask_enable:
    #         atten = encoded_inputs["attention_mask"]
    #         mask_m = torch.rand(atten.shape) < 0.4
    #         mask = torch.where(mask_m, torch.zeros_like(atten), atten)
    #         mask = mask.permute(1, 0).unsqueeze(-1).cuda()
    #         word_embeddings = mask * word_embeddings
    #
    #     rev = self.reverse_variable(word_embeddings)
    #     _, (thoughts, _) = self.lstm(rev)
    #     embeddings_lstm = thoughts[-1]  # (batch, thought_size)
    #     embeddings_gru = self.gru(word_embeddings.permute(1, 0, 2))
    #
    #     if fusion:
    #         out = self.proj_fusion(torch.cat([embeddings_gru + embeddings_lstm, support_feat], dim=1))
    #     else:
    #         out = embeddings_gru + embeddings_lstm
    #
    #     return out

    def forward(self, input, support_feat, fusion=False):
        encoded_inputs = self.tokenizer(input, max_length=39, padding='max_length', truncation=True,
                                        return_tensors="pt")
        sentences = encoded_inputs["input_ids"].cuda()
        sentences = sentences.transpose(0, 1)
        word_embeddings = F.tanh(self.word2embd(sentences))
        if self.constrain and self.training:
            if self.mask_enable:
                atten = encoded_inputs["attention_mask"]
                mask_m = torch.rand(atten.shape) < 0.4
                mask = torch.where(mask_m, torch.zeros_like(atten), atten)
                mask = mask.permute(1, 0).unsqueeze(-1).cuda()
                word_embeddings = mask * word_embeddings

            rev = self.reverse_variable(word_embeddings)
            _, (thoughts, _) = self.lstm(rev)
            embeddings_lstm = thoughts[-1]  # (batch, thought_size)
            embeddings_gru = self.gru(word_embeddings.permute(1, 0, 2))
            text_feat1 = embeddings_lstm + embeddings_gru

            if self.mask_enable:
                atten = encoded_inputs["attention_mask"]
                mask_m = torch.rand(atten.shape) < 0.4
                mask = torch.where(mask_m, torch.zeros_like(atten), atten)
                mask = mask.permute(1, 0).unsqueeze(-1).cuda()
                word_embeddings = mask * word_embeddings

            rev = self.reverse_variable(word_embeddings)
            _, (thoughts, _) = self.lstm(rev)
            embeddings_lstm = thoughts[-1]  # (batch, thought_size)
            embeddings_gru = self.gru(word_embeddings.permute(1, 0, 2))
            text_feat2 = embeddings_lstm + embeddings_gru

            loss2, on_diag2, off_diag2 = self.bt_loss_single(text_feat2, text_feat1)
        else:
            if self.mask_enable and self.training:
                atten = encoded_inputs["attention_mask"]
                mask_m = torch.rand(atten.shape) < 0.4
                mask = torch.where(mask_m, torch.zeros_like(atten), atten)
                mask = mask.permute(1, 0).unsqueeze(-1).cuda()
                word_embeddings = mask * word_embeddings

            rev = self.reverse_variable(word_embeddings)
            _, (thoughts, _) = self.lstm(rev)
            embeddings_lstm = thoughts[-1]  # (batch, thought_size)
            embeddings_gru = self.gru(word_embeddings.permute(1, 0, 2))

        if fusion:
            out = self.proj_fusion(torch.cat([embeddings_gru + embeddings_lstm, support_feat], dim=1))
        else:
            out = embeddings_gru + embeddings_lstm

        if self.constrain and self.training:
            self.constrain_loss['c_loss'] = loss2 * 0.1
            return out
        else:
            return out

    # def bt_loss_single(self, z1, z2):
    #     c = self.bn(z1) @ self.bn(z2).T
    #     on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    #     off_diag = off_diagonal(c).pow_(2).sum()
    #     loss = on_diag + self.lambd * off_diag
    #     return loss, on_diag, off_diag
    def bt_loss_single(self, z1, z2):
        dot_product_mat = torch.mm(z1, torch.transpose(z2, 0, 1))
        len_vec1 = torch.unsqueeze(torch.sqrt(torch.sum(z1 * z1, dim=1)), dim=0)
        len_vec2 = torch.unsqueeze(torch.sqrt(torch.sum(z2 * z2, dim=1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec1, 0, 1), len_vec2)

        cos_sim_mat = dot_product_mat / len_mat
        on_diag = torch.diagonal(cos_sim_mat).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(cos_sim_mat).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        if torch.isnan(loss):
            pdb.set_trace()
        return loss, on_diag, off_diag


class GRUModel(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            output_size,
    ):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)  # output_size 为输出的维度

    def forward(self, input):
        output, _ = self.gru(input)
        output = self.fc(output[:, -1, :])  # 取最后一个时间步的输出

        return output
