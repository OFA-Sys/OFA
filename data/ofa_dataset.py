import logging
import torch.utils.data
from fairseq.data import FairseqDataset

logger = logging.getLogger(__name__)


class OFADataset(FairseqDataset):

    def __len__(self):
        return len(self.dataset)

    def encode_text(self, text, length=None, append_bos=False, append_eos=False):
        s = self.tgt_dict.encode_line(
            line=self.bpe.encode(text),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.bos_item, s])
        if append_eos:
            s = torch.cat([s, self.eos_item])
        return s
