import tensorflow as tf
from utils import iterator_utils
from utils import vocab_utils
import argparse

# If a vocab size is greater than this value, put the embedding on cpu instead
VOCAB_SIZE_THRESHOLD_CPU = 50000

def _get_embed_device(vocab_size):
	if vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
		return "/cpu:0"
	else:
		return "/gpu:0"

def _create_or_load_embed(embed_name, vocab_file, embed_file, vocab_size, embed_size, dtype):
	with tf.device(_get_embed_device(vocab_size)):
		embedding = tf.get_variable(embed_name, [vocab_size, embed_size], dtype)
	return embedding

def create_emb_for_encoder_and_decoder(share_vocab, src_vocab_size, tgt_vocab_size, src_embed_size, tgt_embed_size, dtype=tf.float32, num_partitions=0, src_vocab_file=None, tgt_vocab_file=None, src_embed_file=None, tgt_embed_file=None, scope=None):
	if num_partitions <= 1:
		partitioner = None
	else:
		partitioner = tf.fixed_size_partitioner(num_partitions)

	with tf.variable_scope(scope or "embeddings", dtype=dtype, partitioner=partitioner) as scope:
		vocab_file = src_vocab_file or tgt_vocab_file
		embed_file = src_embed_file or tgt_embed_file

		embedding_encoder = _create_or_load_embed("embedding_share", vocab_file, embed_file, src_vocab_size, src_embed_size, dtype)
		embedding_decoder = embedding_encoder

	return embedding_encoder, embedding_decoder


parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default=None)
parser.add_argument("--tgt", type=str, default=None)
parser.add_argument("--train_prefix", type=str, default=None)
parser.add_argument("--vocab_prefix", type=str, default=None)
parser.add_argument("--embed_prefix", type=str, default=None)
parser.add_argument("--sos", type=str, default="<s>")
parser.add_argument("--eos", type=str, default="</s>")
parser.add_argument("--share_vocab", type=bool, nargs="?", const=True, default=False)

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--random_seed", type=int, default=None)
parser.add_argument("--num_buckets", type=int, default=5)
parser.add_argument("--src_max_len", type=int, default=50)
parser.add_argument("--tgt_max_len", type=int, default=50)

parser.add_argument("--num_units", type=int, default=32)
parser.add_argument("--num_embeddings_partitions", type=int, default=0)

parser.add_argument("--jobid", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=1)

parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--check_special_token", type=bool, default=True)
parser.add_argument("--scope", type=str, default=None)
parser.add_argument("--time_major", type=bool, nargs="?", const=True, default=True)


args = parser.parse_args()

src_file = "%s.%s" % (args.train_prefix, args.src)
tgt_file = "%s.%s" % (args.train_prefix, args.tgt)

src_vocab_file = args.vocab_prefix + "." + args.src
tgt_vocab_file = args.vocab_prefix + "." + args.tgt

#src_embed_file = args.embed_prefix + "." + args.src
#tgt_embed_file = args.embed_prefix + "." + args.tgt
src_embed_file = ""
tgt_embed_file = ""

src_vocab_size, src_vocab_file = vocab_utils.check_vocab(
	src_vocab_file,
	args.out_dir,
	check_special_token=args.check_special_token,
	sos=args.sos,
	eos=args.eos,
	unk=vocab_utils.UNK)

tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(
	tgt_vocab_file,
	args.out_dir,
	check_special_token=args.check_special_token,
	sos=args.sos,
	eos=args.eos,
	unk=vocab_utils.UNK)

graph = tf.Graph()
scope="train"
with graph.as_default(), tf.container(scope):
	src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(src_vocab_file, tgt_vocab_file, args.share_vocab)
	src_dataset = tf.data.TextLineDataset(src_file)
	tgt_dataset = tf.data.TextLineDataset(tgt_file)
	skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

	iterator = iterator_utils.get_iterator(
				src_dataset,
				tgt_dataset,
				src_vocab_table,
				tgt_vocab_table,
				batch_size=args.batch_size,
				sos=args.sos,
				eos=args.eos,
				random_seed=args.random_seed,
				num_buckets=args.num_buckets,
				src_max_len=args.src_max_len,
				tgt_max_len=args.tgt_max_len,
				skip_count=skip_count_placeholder,
				num_shards=args.num_workers,
				shard_index=args.jobid)

# get source
source = iterator.source
if args.time_major:
	source = tf.transpose(source)

embedding_encoder, embedding_decoder = create_emb_for_encoder_and_decoder(share_vocab=args.share_vocab,
																																					src_vocab_size=src_vocab_size,
																																					tgt_vocab_size=tgt_vocab_size,
																																					src_embed_size=args.num_units,
																																					tgt_embed_size=args.num_units,
																																					num_partitions=args.num_embeddings_partitions,
																																					src_vocab_file=src_vocab_file,
																																					tgt_vocab_file=tgt_vocab_file,
																																					src_embed_file=src_embed_file,
																																					tgt_embed_file=tgt_embed_file,
																																					scope=scope,)
print(embedding_encoder)
encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, source)
