import re

def extract_html_content(html_text, search_text):
    pattern = re.compile(re.escape(search_text) + r'(.*?)</html>', re.DOTALL)
    match = pattern.search(html_text)
    if match:
        return search_text + match.group(1) + '</html>'
    else:
        return "No match found."

def extract_html_content_siri(html_text):
    match = re.search(r'^.*</html>', html_text, re.DOTALL)
    if match:
        return match.group()# + '</html>'
    else:
        return "No match found."


text1 = """Text: A page for an introduction to the product. The page has three groups of information. Each group contains a title , a description, and a button for users to click to get further information.
<html>
<body>
<div class="canvas" style="left: 0px; top: 0px; width: 120px; height: 120px"></div>
<div class="title" style="left: 5px; top: 9px; width: 50px; height: 10px"></div>
<div class="title" style="left: 63px; top: 9px; width: 18px; height: 3px"></div>
<div class="title" style="left: 88px; top: 9px; width: 18px; height: 6px"></div>
<div class="description" style="left: 63px; top: 14px; width: 16px; height: 9px"></div>
<div class="description" style="left: 88px; top: 17px; width: 18px; height: 9px"></div>
<div class="description" style="left: 5px; top: 20px; width: 47px; height: 9px"></div>
<div class="button" style="left: 63px; top: 28px; width: 17px; height: 5px"></div>
<div class="button" style="left: 88px; top: 31px; width: 18px; height: 7px"></div>
<div class="button" style="left: 5px; top: 33px; width: 18px; height: 4px"></div>
</body>
</html>
"""
text2 = """<html>
<body>
<div class="canvas" style="left: 0px; top: 0px; width: 120px; height: 120px"></div>
<div class="title" style="left: 6px; top: 12px; width: 107px; height: 2px"></div>
<div class="description" style="left: 6px; top: 17px; width: 107px; height: 3px"></div>
<div class="input" style="left: 27px; top: 27px; width: 48px; height: 5px"></div>
<div class="button" style="left: 77px; top: 29px; width: 12px; height: 2px"></div>
</body>
</html>
"""
print(extract_html_content_siri(text2))
