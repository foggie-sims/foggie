from docutils import nodes
from docutils.parsers.rst import \
     Directive, \
     directives

########################################################################
# Taken from http://countergram.com/youtube-in-rst (MIT License)
########################################################################

CODE = """\
<object type="application/x-shockwave-flash"
        width="%(width)s"
        height="%(height)s"
        class="youtube-embed"
        data="http://www.youtube.com/v/%(yid)s">
    <param name="movie" value="http://www.youtube.com/v/%(yid)s"></param>
    <param name="wmode" value="transparent"></param>%(extra)s
</object>
"""

PARAM = """\n    <param name="%s" value="%s"></param>"""

def setup(app):
    app.add_node(youtube)
    app.add_directive('youtube', YouTubeDirective)

class youtube(nodes.General, nodes.Element):
    pass

class YouTubeDirective(Directive):
    has_content = True
    node_class = None
    def run(self):
        """ Restructured text extension for inserting youtube embedded videos """
        if len(self.content) == 0:
            return
        string_vars = {
            'yid': self.content[0],
            'width': 425,
            'height': 344,
            'extra': ''
            }
        extra_args = self.content[1:] # Because self.content[0] is ID
        extra_args = [ea.strip().split("=") for ea in extra_args] # key=value
        extra_args = [ea for ea in extra_args if len(ea) == 2] # drop bad lines
        extra_args = dict(extra_args)
        if 'width' in extra_args:
            string_vars['width'] = extra_args.pop('width')
        if 'height' in extra_args:
            string_vars['height'] = extra_args.pop('height')
        if extra_args:
            params = [PARAM % (key, extra_args[key]) for key in extra_args]
            string_vars['extra'] = "".join(params)
        return [nodes.raw('', CODE % (string_vars), format='html')]
