import configobj
import yaml
from configobj import ConfigObj


class GlobalConfigYaml(object):
    def __init__(self, cfg_path='modules_params.yaml', **kwargs):
        self.cfg_path = cfg_path

        def join_and_eval_constructor(loader, node):
            seq = loader.construct_sequence(node)
            str_seq = ''.join([str(i) for i in seq])
            return eval(str_seq)

        yaml.add_constructor('!eval', join_and_eval_constructor)
        with open(cfg_path) as f:
            config = f.read()
        anchors = self.anchors(kwargs)
        self.obj = yaml.load(anchors + config)

    def anchors(self, kwargs):
        base_str = '{anchor}: &{anchor} {value}\n'
        text = ''
        for k, v in kwargs.items():
            text += base_str.format(anchor=k, value=v)
        return text


class GlobalConfig(ConfigObj):
    def __init__(self, infile='modules_params.ini', user_values=None, **kwargs):
        usr = ConfigObj(infile=user_values)
        defaults = ConfigObj(infile)
        defaults.merge(usr)
        super(GlobalConfig, self).__init__(infile=defaults.dict(), interpolation='template', **kwargs)

    def eval_dict(self):
        return self.recursive_eval(self.dict())

    @staticmethod
    def recursive_eval(section):
        def try_eval(obj):
            try:
                return configobj.unrepr(obj)
            except ValueError:
                pass
            try:
                return eval(obj)
            except Exception as e:
                print("Object: {} isn't literal but can't evaluate expression".format(obj))

        for k, value in section.items():
            if isinstance(value, dict):
                section[k] = GlobalConfig.recursive_eval(value)
                continue

            section[k] = try_eval(value)
        return section
