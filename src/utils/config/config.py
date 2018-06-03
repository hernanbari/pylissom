import configobj
import os
import yaml
from configobj import ConfigObj

here = os.path.abspath(os.path.dirname(__file__))


class EvalConf(object):
    def eval_dict(self):
        raise NotImplementedError


class EvalConfigYaml(EvalConf):
    def __init__(self, infile=os.path.join(here, 'modules_params.yaml'), user_values=None):
        self.infile = infile

        def join_and_eval_constructor(loader, node):
            seq = loader.construct_sequence(node)
            str_seq = ''.join([str(i) for i in seq])
            return eval(str_seq)

        yaml.add_constructor('!eval', join_and_eval_constructor)
        with open(infile) as f:
            config = f.read()
        anchors = self.anchors(user_values)
        self.conf = yaml.load(anchors + config)

    def eval_dict(self):
        return self.conf

    def anchors(self, kwargs):
        base_str = '{anchor}: &{anchor} {value}\n'
        text = ''
        for k, v in kwargs.items():
            text += base_str.format(anchor=k, value=v)
        return text


class EvalConfigObj(ConfigObj, EvalConf):
    def __init__(self, infile=None, user_values=None, **kwargs):
        infile = os.path.join(here, 'modules_params.ini') if infile is None else infile
        usr = ConfigObj(infile=user_values)
        defaults = ConfigObj(infile)
        defaults.merge(usr)
        super(EvalConfigObj, self).__init__(infile=defaults.dict(), interpolation='template', **kwargs)

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
                section[k] = EvalConfigObj.recursive_eval(value)
                continue

            section[k] = try_eval(value)
        return section


def global_config(conf_obj=True, *args, **kwargs):
    return EvalConfigObj(*args, **kwargs) if conf_obj else EvalConfigYaml(*args, **kwargs)

