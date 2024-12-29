from markitdown import MarkItDown

md = MarkItDown()
file_name = "2024.07 Panorama - hub overview V2 copy 13u53"
result = md.convert(f"{file_name}.xlsx")

# write result.text_content to a markdown file
with open(f"{file_name}.md", "w") as file:
    file.write(result.text_content)


print(result.text_content)

a=2