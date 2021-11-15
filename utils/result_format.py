import os
import os.path as osp
import numpy as np
import json
import zipfile


class ResultFormat(object):
    def __init__(self, data_type, result_path):
        self.data_type = data_type
        self.result_path = result_path

        if osp.isfile(result_path):
            os.remove(result_path)

        if result_path.endswith('.zip'):
            result_path = result_path.replace('.zip', '')

        if not osp.exists(result_path):
            os.makedirs(result_path)

    def write_result(self, img_name, outputs):
        if 'IC15' in self.data_type:
            self._write_result_ic15(img_name, outputs)
        elif 'TT' in self.data_type:
            self._write_result_tt(img_name, outputs)
        elif 'CTW' in self.data_type:
            self._write_result_ctw(img_name, outputs)
        elif 'MSRA' in self.data_type:
            self._write_result_msra(img_name, outputs)
        elif 'BENCHMARK' in self.data_type:
            self._write_result_benchmark(img_name, outputs)

    def _write_result_benchmark(self, image_name, outputs):
        bboxes = outputs['bboxes']
        words = outputs['words']

        results = []
        for i, bbox in enumerate(bboxes):
            if words[i] is not None:
                values = [round(v) for v in bbox]
                poly = np.array(values, dtype=np.float32).reshape(-1, 2).tolist()
                transcript = words[i]
                score = 1
                result = {'poly': poly, 'transcript': transcript, 'score': score}
                results.append(result)

        file_name = '%s.json' % image_name
        file_path = osp.join(self.result_path, file_name)
        f = open(file_path, "w", encoding="utf-8")
        json.dump({'image_name': image_name, 'preds': results}, f)
        f.close()

    def _write_result_ic15(self, img_name, outputs):
        assert self.result_path.endswith(
            '.zip'), 'Error: ic15 result should be a zip file!'

        tmp_folder = self.result_path.replace('.zip', '')

        bboxes = outputs['bboxes']
        words = None
        if 'words' in outputs:
            words = outputs['words']

        lines = []
        for i, bbox in enumerate(bboxes):
            values = [int(v) for v in bbox]
            if words is None:
                line = '%d,%d,%d,%d,%d,%d,%d,%d\n' % tuple(values)
                lines.append(line)
            elif words[i] is not None:
                line = '%d,%d,%d,%d,%d,%d,%d,%d' % tuple(
                    values) + ',%s\n' % words[i]
                lines.append(line)

        file_name = 'res_%s.txt' % img_name
        file_path = osp.join(tmp_folder, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

        z = zipfile.ZipFile(self.result_path, 'a', zipfile.ZIP_DEFLATED)
        z.write(file_path, file_name)
        z.close()

    def _write_result_tt(self, image_name, outputs):
        bboxes = outputs['bboxes']

        lines = []
        for i, bbox in enumerate(bboxes):
            bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1)
            values = [int(v) for v in bbox]
            line = '%d' % values[0]
            for v_id in range(1, len(values)):
                line += ',%d' % values[v_id]
            line += '\n'
            lines.append(line)

        file_name = '%s.txt' % image_name
        file_path = osp.join(self.result_path, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

    def _write_result_ctw(self, image_name, outputs):
        bboxes = outputs['bboxes']

        lines = []
        for i, bbox in enumerate(bboxes):
            bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1)
            values = [int(v) for v in bbox]
            line = '%d' % values[0]
            for v_id in range(1, len(values)):
                line += ',%d' % values[v_id]
            line += '\n'
            lines.append(line)

        file_name = '%s.txt' % image_name
        file_path = osp.join(self.result_path, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

    def _write_result_msra(self, image_name, outputs):
        bboxes = outputs['bboxes']
    
        lines = []
        for b_idx, bbox in enumerate(bboxes):
            values = [int(v) for v in bbox]
            line = '%d' % values[0]
            for v_id in range(1, len(values)):
                line += ', %d' % values[v_id]
            line += '\n'
            lines.append(line)
    
        file_name = '%s.txt' % image_name
        file_path = osp.join(self.result_path, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)
