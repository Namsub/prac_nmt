import tensorflow as tf

from tensorflow.python.client import timeline
from utils import iterator_utils
from utils import vocab_utils
import argparse

# If a vocab size is greater than this value, put the embedding on cpu instead
VOCAB_SIZE_THRESHOLD_CPU = 50000

# PATH
PATH = "/home/titanxp/prac_nmt"

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

def _single_cell(unit_type, num_units, forget_bias, dropout, mode, residual_connection=False, device_str=None, residual_fn=None):
	"""Create an instance of a single RNN cell."""
	dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

	# Cell Type
	if unit_type == "lstm":
		single_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=forget_bias)
	elif unit_type == "gru":
		single_cell = tf.contrib.rnn.GRUCell(num_units)
	elif unit_type == "layer_norm_lstm":
		single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units, forget_bias=forget_bias, layer_norm=True)
	elif unit_type == "nas":
		single_cell = tf.contrib.rnn.NASCell(num_units)
	else:
		raise ValueError("Unknown unit type %s!" % unit_type)

	# Dropout (= 1 - keep_prob)
	if dropout > 0.0:
		single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))

	# Residual
	if residual_connection:
		single_cell = tf.contrib.rnn.ResidualWrapper(single_cell, residual_fn=residual_fn)

	# Device Wrapper
	if device_str:
		single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
	
	return single_cell

def get_device_str(device_id, num_gpus):
	if num_gpus == 0:
		return "/cpu:0"
	device_str_output = "/gpu:%d" % (device_id % num_gpus)
	return device_str_output

def _cell_list(unit_type, num_units, num_layers, num_residual_layers, forget_bias, dropout, mode, num_gpus, base_gpu=0, single_cell_fn=None, residual_fn=None):
	if not single_cell_fn:
		single_cell_fn = _single_cell
	
	# Multi-GPU
	cell_list = []
	for i in range(num_layers):
		single_cell = single_cell_fn(
						unit_type=unit_type,
						num_units=num_units,
						forget_bias=forget_bias,
						dropout=dropout,
						mode=mode,
						residual_connection=(i >= num_layers - num_residual_layers),
						device_str=get_device_str(i + base_gpu, num_gpus),
						residual_fn=residual_fn
		)
		cell_list.append(single_cell)

	return cell_list

def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers, forget_bias, dropout, mode, num_gpus, base_gpu=0, single_cell_fn=None):
	cell_list = _cell_list(unit_type=unit_type,
												num_units=num_units,
												num_layers=num_layers,
												num_residual_layers=num_residual_layers,
												forget_bias=forget_bias,
												dropout=dropout,
												mode=mode,
												num_gpus=num_gpus,
												base_gpu=base_gpu,
												single_cell_fn=single_cell_fn)
	if len(cell_list) == 1:
		return cell_list[0]
	else:	# Multi layers
		return tf.contrib.rnn.MultiRNNCell(cell_list)

def _build_encoder_cell(args, num_layers, num_residual_layers, base_gpu=0):
	return create_rnn_cell(
			unit_type=args.unit_type,
			num_units=args.num_units,
			num_layers=num_layers,
			num_residual_layers=num_residual_layers,
			forget_bias=args.forget_bias,
			dropout=args.dropout,
			num_gpus=args.num_gpus,
			mode=tf.contrib.learn.ModeKeys.TRAIN,
			base_gpu=base_gpu,
			single_cell_fn=None)
	
def _build_bidirectional_rnn(inputs, sequence_length, dtype, args, num_bi_layers, num_bi_residual_layers, base_gpu=0):
	# Construct forward and backward cells
	fw_cell = _build_encoder_cell(args, num_bi_layers, num_bi_residual_layers, base_gpu=base_gpu)
	bw_cell = _build_encoder_cell(args, num_bi_layers, num_bi_residual_layers, base_gpu=(base_gpu + num_bi_layers))

	bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, dtype=dtype, sequence_length=sequence_length, time_major=args.time_major, swap_memory=True)
	return tf.concat(bi_outputs, -1), bi_state

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default=None)
parser.add_argument("--tgt", type=str, default=None)
parser.add_argument("--train_prefix", type=str, default=None)
parser.add_argument("--vocab_prefix", type=str, default=None)
parser.add_argument("--embed_prefix", type=str, default=None)
parser.add_argument("--sos", type=str, default="<s>")
parser.add_argument("--eos", type=str, default="</s>")
parser.add_argument("--share_vocab", type=bool, nargs="?", const=True, default=False)

parser.add_argument("--random_seed", type=int, default=None)
parser.add_argument("--num_buckets", type=int, default=5)
parser.add_argument("--src_max_len", type=int, default=50)
parser.add_argument("--tgt_max_len", type=int, default=50)

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_encoder_layers", type=int, default=None)
parser.add_argument("--num_units", type=int, default=32)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--num_embeddings_partitions", type=int, default=0)
parser.add_argument("--unit_type", type=str, default="lstm", help="lstm | gru | layer_norm_lstm | nas")
parser.add_argument("--forget_bias", type=float, default=0.2)

parser.add_argument("--jobid", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--num_gpus", type=int, default=1)

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

#graph = tf.Graph()
scope="train"
#with graph.as_default(), tf.container(scope):
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
#print(embedding_encoder)
encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, source)
#print(encoder_emb_inp)

num_layers = args.num_layers
#num_encoder_layers = (args.num_encoder_layers or args.num_layers)
#num_residual_layers = args.num_encoder_residual_layers
num_residual_layers = 0

num_bi_layers = int(num_layers / 2)
num_bi_residual_layers = int(num_residual_layers / 2)

dtype = tf.float32
encoder_outputs, bi_encoder_state = _build_bidirectional_rnn(
																			inputs=encoder_emb_inp, 
																			sequence_length=iterator.source_sequence_length,
																			dtype=dtype,
																			args=args,
																			num_bi_layers=num_bi_layers,
																			num_bi_residual_layers=num_bi_residual_layers)

init_op = tf.initialize_all_variables()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
with tf.Session() as sess:
	sess.run(init_op)
	sess.run(tf.tables_initializer())
	sess.run(iterator.initializer, feed_dict={skip_count_placeholder: 0})
	sess.run(encoder_outputs, options=run_options, run_metadata=run_metadata)
	sess.run(bi_encoder_state, options=run_options, run_metadata=run_metadata)

	tl = timeline.Timeline(step_stats=run_metadata.step_stats)
	ctf = tl.generate_chrome_trace_format(show_memory=True)

	trace_name = "trace_outputs_state.json"
	trace_file = PATH + '/' + trace_name
	with open(trace_file, 'w') as f:
		f.write(ctf)
#print(encoder_outputs, bi_encoder_state)
