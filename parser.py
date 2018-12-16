# -*- coding: utf-8 -*-

r"""A JSON parser using funcparserlib.

The parser is based on [the JSON grammar][1].

  [1]: http://tools.ietf.org/html/rfc4627
"""

import sys
import os
import re
import logging
from re import VERBOSE
from pprint import pformat
from funcparserlib.lexer import make_tokenizer, Token, LexerError
from funcparserlib.parser import (some, a, maybe, many, finished, skip,
                                  forward_decl, NoParseError)

from pprint import pprint as pp 


ENCODING = 'UTF-8'
regexps = {
    'escaped': r'''
        \\                                  # Escape
          ((?P<standard>["\\/bfnrt])        # Standard escapes
        | ((?P<unicode>[0-9A-Fa-f]{4})))   # uXXXX
        ''',
    'unescaped': r'''
        [^"\\]                              # Unescaped: avoid ["\\]
        ''',
}
re_esc = re.compile(regexps['escaped'], VERBOSE)
"""
Variant::
    is_master : bool
    track_inventory : bool(255 'th,is') , , 
    is_master : bool  n='2'

Product::
    is_master : bool
    track_inventory : bool
    sku : char max_length=255
    position : int(244, 'this' cool)  a=none b=false c=true 

Other::
    sku : char max_length=255
    position : int a:none b:false c:true
    is_master : bool
    track_inventory : bool
    weight: decimal  max_digits=8 m=8.0
    product_id: fk null=false on_delete=models.PROTECT 
"""

parse_text = ''
with open('grammar') as fil_:
    parse_text = fil_.read()

class Field(object):
    def __init__(self, definition):
        #('fieldname', ('fieldtype', ['args', "123"]), [kwargs=true]),
        self._name = definition[0]
        self._type = definition[1][0]
        self._args = definition[1][1] or []  # may be none type
        self._kwargs = definition[2] or []

    def camel_case(self, name):
        splits = name.split('_')
        return ''.join(word.capitalize() for word in splits)

    def get_name(self):
        if self._type in ['fk', 'm2m', 'o2o'] and self._name.endswith('_id'):
            return self._name[:-len('_id')]
        return self._name

    def get_type(self):
        if self._type == 'fk': return 'models.ForeignKey'
        elif self._type == 'm2m': return 'models.ManyToManyField'
        elif self._type == 'o2o': return 'models.OneToOneField'
        elif self._type == 'bool': return 'models.BooleanField'
        elif self._type == 'char': return 'models.CharField'
        elif self._type == 'date': return 'models.DateField'
        elif self._type == 'datetime': return 'models.DateTimeField'
        elif self._type == 'decimal': return 'models.DecimalField'
        elif self._type == 'duration': return 'models.DurationField'
        elif self._type == 'email': return 'models.EmailField'
        elif self._type == 'float': return 'models.FloatField'
        elif self._type == 'int': return 'models.IntegerField'
        elif self._type == 'slug': return 'models.SlugField'
        elif self._type == 'text': return 'models.TextField'
        elif self._type == 'time': return 'models.TimeField'
        elif self._type == 'url': return 'models.URLField'
        elif self._type == 'uuid': return 'models.UUIDField'

        else: 
            raise Exception('No type %s for %s defined'%(self._type, self._name))

    def get_params(self):
        referred_model = ''
        args = ', '.join(self._args)
        kwargs = ', '.join('%s=%s'%(x,y) for x,y in self._kwargs)
        if self._type in ['fk', 'm2m', 'o2o'] and not self._args: 
            referred_model = "'%s'"%self.camel_case(self.get_name())

        return ', '.join(x for x in [referred_model, args, kwargs] if x) 

    def to_string(self):
        fieldname = self.get_name()
        fieldtype = self.get_type()
        fieldparams = self.get_params()
        return '%s = %s(%s)'%(fieldname, fieldtype, fieldparams)
        
    def __repr__(self):
        return self.to_string()

import re
def tokenize(str):
    """str -> Sequence(Token)"""
    specs = [
    ('COMMENT', (r'//.*',)),
    ('COMMENT', (r'/\*(.|[\r\n])*?\*/', re.MULTILINE)),
    ('NL', (r'[\r\n]+',)),
    ('SPACE', (r'[ \t\r\n]+',)),
    ('REAL', (r'[0-9]+\.[0-9]*([Ee][+\-]?[0-9]+)*',)),
    ('INT', (r'[0-9]+',)),
    ('INT', (r'\$[0-9A-Fa-f]+',)),
    ('OP', (r'(::)|(:)|(-)|(=)|(\()|(\))',)),
    ('FIELD', (r'(bool|char|date|datetime|decimal|duration|email|float|int|slug|text|time|url|uuid|fk|m2m|o2o)',)),
 
    ('NONE', (r'none',)),
    ('FALSE', (r'false',)),
    ('TRUE', (r'true',)),
    ('NAME', (r'([A-Za-z_.][A-Za-z_0-9.]*)',)),
    #('OP', (r'(\.\.)|(<>)|(<=)|(>=)|(:=)|[;,=\(\):\[\]\.+\-<>\*/@\^]',)),
    ('STRING', (r"'([^']|(''))*'",)),
    ('ESCAPES', (r',',)),
]


    useless = ['SPACE', 'NL', 'COMMENT', 'ESCAPES']

    t = make_tokenizer(specs)
    ret  = [x for x in t(str) if x.type not in useless]

    return ret 


def parse(seq):
    """Sequence(Token) -> object"""
    const = lambda x: lambda _: x
    tokval = lambda x: x.value
    toktype = lambda t: some(lambda x: x.type == t) >> tokval
    op = lambda s: a(Token('OP', s)) >> tokval
    op_ = lambda s: skip(op(s))
    n = lambda s: a(Token('NAME', s)) >> tokval

    def make_array(n):
        if n is None:
            return []
        else:
            return [n[0]] + n[1]

    def make_object(n):
        return dict(make_array(n))

    def make_int(n):
        return '%s'%int(n)

    def make_real(n):
        return '%s'%float(n)

    def unescape(s):
        std = {
            '"': '"', '\\': '\\', '/': '/', 'b': '\b', 'f': '\f',
            'n': '\n', 'r': '\r', 't': '\t',
        }

        def sub(m):
            if m.group('standard') is not None:
                return std[m.group('standard')]
            else:
                return unichr(int(m.group('unicode'), 16))

        return re_esc.sub(sub, s)

    def make_string(n):
        return n
        #return unescape(n[1:-1])

    def make_all_models(models):
        return dict(models)

#   all_attrs = []
#        for i in attrs:
#            attr = i[0] 
#            if attr not in all_attrs:
#                all_attrs.append(attr)
#            else:
#                raise Exception('Attribute %s is already defined in class'%attr)

    def make_fields(n):
        #return dict(n)
        return Field(n)


    def make_params(n):
        return n 

    null = toktype('NONE') >> const("None")
    true = toktype('TRUE') >> const("True")
    false = toktype('FALSE') >> const("False")
    number = toktype('INT') >> make_int
    real = toktype('REAL') >> make_real
    string = toktype('STRING') >> make_string
    value = forward_decl()
    name = toktype('NAME') 
    field = toktype('FIELD') + maybe(op_('(') + many(value) + op_(')'))  >> tuple
    member = string + op_(':') + value >> tuple
    attrs = forward_decl()
    params  = forward_decl()

    models = many(
                name + op_('::') + 
                many(attrs) 
            ) >> make_all_models

    attrs.define(
                name + op_(':') + field + many(params) >> make_fields
            )

    params.define(
                name + op_('=') + value >> tuple
            )

    value.define(
        null
        | true
        | false
        | name 
        | number
        | real
        | string)
    parser_text = models
    parser_file = parser_text + skip(finished)
    return parser_file.parse(seq)



from jinja2 import Template

def loads(s):
    """str -> object"""
    tokens = tokenize(s)
    return parse(tokens)


logging.basicConfig(level=logging.DEBUG)
try:
    tree = loads(parse_text)
    with open('schema') as file_:
        template = Template(file_.read())
    print(template.render(data=tree))

        
except (NoParseError, LexerError) as  e:
    msg = ('syntax error: %s' % e).encode(ENCODING)
    print >> sys.stderr, msg
    sys.exit(1)


