import xml.etree.ElementTree as ET
import pathlib

TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
%s
</routes>"""

for i, route in enumerate(ET.parse('routes_training.xml').getroot()):
    pathlib.Path('separated/route_%02d.xml' % i).write_text(TEMPLATE % ET.tostring(route))
