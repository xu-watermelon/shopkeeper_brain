# knowledge/processor/import_process/nodes/entry_node.py

"""
入口节点

检测文件类型并设置处理标志
"""

import json
from pathlib import Path
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError


class EntryNode(BaseNode):
    """
    入口节点

    根据输入文件的扩展名设置相应的处理标志，
    决定后续流程走 PDF 转换分支还是直接处理 MD 分支。
    """

    name = "entry"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        处理文件类型的检测

        Args:
            state: ImportGraphState 该节点处理之前的节点状态

        Returns:
            ImportGraphState：该节点处理之后的节点状态
        """
        # 1. 获取导入的文件路径以及文件所在的目录
        self.log_step("Step1", "[获取文件路径]")
        file_dir = state.get('file_dir') #上传PDF后存储的目录 D:\test_data,以及后续用这个目录存储生成的MD文件
        import_file_path = state.get('import_file_path') #导入文件的完整路径及文件名称 D:\test_data\万用表的使用.pdf|md

        # 2. 简单校验一下 文件路径以及所在目录
        self.log_step("Step2", "[检测文件路径]")
        if not file_dir or not import_file_path:
            raise ValidationError("文件目录或者文件不存在", self.name)

        # 3. 使用标准的Path对象操作文件逻辑
        path = Path(import_file_path)

        # 4. 获取上传文件的后缀
        suffix = path.suffix.lower()

        # 5. 判断文件的后缀
        if suffix == '.pdf':
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = import_file_path
        elif suffix == '.md':
            state['is_md_read_enabled'] = True
            state['md_path'] = import_file_path
        else:
            self.logger.debug(f"文件类型{suffix}不支持")
            raise ValidationError(f"文件类型{suffix}不支持")

        # 6. 获取文件的标题名
        file_title = path.stem
        state['file_title'] = file_title

        # 7. 返回state
        return state


# ================================================================== #
#                        测试                                        #
# ================================================================== #

if __name__ == '__main__':
    setup_logging()

    # 方式：直接实例该节点对象，调用 process 方法
    # 1. 构建该节点需要的 state
    test_entry_state = {
        "file_dir": r"D:\test_data",
        "import_file_path": r"D:\test_data\万用表的使用.pdf"
    }

    # 2. 实例 EntryNode 节点
    entry_node = EntryNode()

    # 3. 调用 process 方法
    processed_state = entry_node(test_entry_state)

    # 序列化打印
    print(json.dumps(processed_state, ensure_ascii=False, indent=4))