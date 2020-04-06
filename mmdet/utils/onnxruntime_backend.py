import onnx
import onnxruntime
from onnx import helper, shape_inference
from onnx.utils import polish_model

from mmdet.models import build_detector


class ONNXModel(object):

    def __init__(self, model_file_path, cfg=None, classes=None):
        self.device = onnxruntime.get_device()
        self.model = onnx.load(model_file_path)
        # self.model = polish_model(self.model)
        self.classes = classes
        self.pt_model = None
        if cfg is not None:
            self.pt_model = build_detector(
                cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
            if classes is not None:
                self.pt_model.CLASSES = classes

        self.sess_options = onnxruntime.SessionOptions()
        # self.sess_options.enable_profiling = False

        self.session = onnxruntime.InferenceSession(
            self.model.SerializeToString(), self.sess_options)
        self.input_names = []
        self.output_names = []
        for input in self.session.get_inputs():
            self.input_names.append(input.name)
        for output in self.session.get_outputs():
            self.output_names.append(output.name)

    def show(self, data, result, dataset=None, score_thr=0.3, wait_time=0):
        if self.pt_model is not None:
            self.pt_model.show_result(
                data, result, dataset=dataset, score_thr=score_thr, wait_time=wait_time)

    def add_output(self, output_ids):
        if not isinstance(output_ids, (tuple, list, set)):
            output_ids = [
                output_ids,
            ]

        inferred_model = shape_inference.infer_shapes(self.model)
        all_blobs_info = {
            value_info.name: value_info
            for value_info in inferred_model.graph.value_info
        }

        extra_outputs = []
        for output_id in output_ids:
            value_info = all_blobs_info.get(output_id, None)
            if value_info is None:
                print('WARNING! No blob with name {}'.format(output_id))
                extra_outputs.append(
                    helper.make_empty_tensor_value_info(output_id))
            else:
                extra_outputs.append(value_info)

        self.model.graph.output.extend(extra_outputs)
        self.output_names.extend(output_ids)
        self.session = onnxruntime.InferenceSession(
            self.model.SerializeToString(), self.sess_options)

    def __call__(self, inputs, *args, **kwargs):
        if not isinstance(inputs, dict):
            if len(self.input_names) == 1 and not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            inputs = dict(zip(self.input_names, inputs))
        outputs = self.session.run(None, inputs, *args, **kwargs)
        outputs = dict(zip(self.output_names, outputs))
        return outputs
