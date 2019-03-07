import os
import sys

from tensor2tensor.bin import t2t_trainer


def problem_args(problem_name):

  args = [
    '--generate_data',
    '--model=transformer',
    '--hparams_set=transformer_librispeech_v1',
    '--problem=%s' % problem_name,
    '--data_dir=/tmp/refactor_test/problems/%s/data' % problem_name,
    '--tmp_dir=/tmp/refactor_test/problems/%s/tmp' % problem_name,
    '--output_dir=/tmp/refactor_test/models/%s/data' % problem_name,
    '--hparams=batch_shuffle_size=0,batch_size=1000000'
  ]

  return args


def main():

  sys.argv += problem_args('librispeech_clean_small')
  # sys.argv += problem_args('common_voice')

  t2t_trainer.main(None)

  print('All done.')


if __name__ == '__main__':
  main()
