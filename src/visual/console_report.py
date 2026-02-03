from src.arch.perf_calculator import ModelPerformance
from src.visual.report_base import ReportFormatter

class ConsoleReportFormatter(ReportFormatter):
    """控制台输出格式化器 - 美化的表格格式"""
    
    @staticmethod
    def _display_width(text: str) -> int:
        """计算文本的显示宽度（考虑中文字符占2个宽度）"""
        width = 0
        for char in text:
            if ord(char) >= 0x4E00 and ord(char) <= 0x9FFF:  # 中文字符范围
                width += 2
            else:
                width += 1
        return width
    
    @staticmethod
    def _pad_string(text: str, width: int, align: str = 'left') -> str:
        """根据显示宽度填充字符串"""
        display_len = ConsoleReportFormatter._display_width(text)
        padding = width - display_len
        if padding <= 0:
            return text
        if align == 'left':
            return text + ' ' * padding
        else:
            return ' ' * padding + text
    
    def format(self, model_perf: ModelPerformance) -> None:
        """
        格式化并打印性能报告到控制台
        
        Args:
            model_perf: 模型性能指标
        """
        all_rows = self._collect_data(model_perf)
        
        # 计算每列的最大宽度（考虑中文字符）
        if all_rows:
            col_widths = {}
            # 文本列
            col_widths['name'] = max(
                max(self._display_width(row['name']) for row in all_rows),
                self._display_width('算子名称')
            ) + 2
            col_widths['type'] = max(
                max(self._display_width(row['type']) for row in all_rows),
                self._display_width('类型')
            ) + 1
            # 数值列
            col_widths['m'] = max(max(len(str(row['m'])) for row in all_rows), len('m')) + 1
            col_widths['n'] = max(max(len(str(row['n'])) for row in all_rows), len('n')) + 1
            col_widths['k'] = max(max(len(str(row['k'])) for row in all_rows), len('k')) + 1
            col_widths['batch'] = max(max(len(str(row['batch'])) for row in all_rows), len('batch')) + 1
            col_widths['layers'] = max(max(len(str(row['layers'])) for row in all_rows), len('layers')) + 1
            col_widths['in_dtype'] = max(
                max(self._display_width(row['in_dtype']) for row in all_rows),
                self._display_width('输入')
            ) + 1
            col_widths['out_dtype'] = max(
                max(self._display_width(row['out_dtype']) for row in all_rows),
                self._display_width('输出')
            ) + 1
            col_widths['weight_dtype'] = max(
                max(self._display_width(row['weight_dtype']) for row in all_rows),
                self._display_width('权重')
            ) + 1
            # 浮点数列（宽度足以显示格式化后的值）
            col_widths['compute'] = max(len('计算(us)'), 10) + 1
            col_widths['memory'] = max(len('内存(us)'), 10) + 1
            col_widths['transfer'] = max(len('传输(us)'), 10) + 1
            col_widths['op_time_single_layer'] = max(len('单层理论延时(us)'), 15) + 1
            col_widths['total'] = max(len('总时间(ms)'), 10) + 1
            col_widths['percent'] = max(len('占比(%)'), 8) + 1
        else:
            col_widths = {
                'name': 20, 'type': 10, 'm': 6, 'n': 6, 'k': 6,
                'batch': 8, 'layers': 8, 'in_dtype': 8, 'out_dtype': 8,
                'weight_dtype': 8, 'compute': 12, 'memory': 12, 'transfer': 12, 'op_time_single_layer': 12,
                'total': 12, 'percent': 10
            }
        
        # 计算总宽度（考虑分隔符）
        total_width = sum(col_widths.values()) + 16  # 16 = 分隔符号的宽度
        
        # 打印表头
        print()
        print("┌" + "─" * (total_width - 2) + "┐")
        header_text = f"性能分析报告: {model_perf.model_name} ({model_perf.forward_mode})"
        padding = total_width - self._display_width(header_text) - 4
        print(f"│ {header_text}" + " " * max(padding, 1) + " │")
        print("├" + "─" * (total_width - 2) + "┤")
        
        # 打印列标题
        headers = [
            ('name', '算子名称'),
            ('type', '类型'),
            ('m', 'm'),
            ('n', 'n'),
            ('k', 'k'),
            ('batch', 'batch'),
            ('layers', 'layers'),
            ('in_dtype', '输入'),
            ('out_dtype', '输出'),
            ('weight_dtype', '权重'),
            ('compute', '计算(us)'),
            ('memory', '内存(us)'),
            ('transfer', '传输(us)'),
            ('op_time_single_layer', '单层理论延时(us)'),
            ('total', '总时间(ms)'),
            ('percent', '占比(%)'),
        ]
        
        header_line = "│ "
        for key, label in headers:
            width = col_widths[key]
            if key in ['compute', 'memory', 'transfer', 'total', 'percent']:
                # 数值列右对齐
                padded = self._pad_string(label, width, 'right')
            else:
                # 文本列左对齐
                padded = self._pad_string(label, width, 'left')
            header_line += f"{padded} │ "
        print(header_line)
        print("├" + "─" * (total_width - 2) + "┤")
        
        # 打印数据行
        for row in all_rows:
            line = "│ "
            for key, _ in headers:
                width = col_widths[key]
                value = row[key]
                
                if key in ['compute', 'memory', 'transfer']:
                    # 浮点数，3位小数
                    formatted = f"{value:.3f}"
                    padded = self._pad_string(formatted, width, 'right')
                    line += f"{padded} │ "
                elif key == 'op_time_single_layer':
                    # 单层理论延时，3位小数
                    formatted = f"{value:.3f}"
                    padded = self._pad_string(formatted, width, 'right')
                    line += f"{padded} │ "
                elif key == 'total':
                    # 总时间，3位小数
                    formatted = f"{value:.3f}"
                    padded = self._pad_string(formatted, width, 'right')
                    line += f"{padded} │ "
                elif key == 'percent':
                    # 百分比，2位小数
                    formatted = f"{value:.2f}"
                    padded = self._pad_string(formatted, width, 'right')
                    line += f"{padded} │ "
                elif key in ['m', 'n', 'k', 'batch', 'layers']:
                    # 整数列
                    formatted = str(value)
                    padded = self._pad_string(formatted, width, 'right')
                    line += f"{padded} │ "
                else:
                    # 文本列
                    padded = self._pad_string(str(value), width, 'left')
                    line += f"{padded} │ "
            print(line)
        
        print("├" + "─" * (total_width - 2) + "┤")
        
        # 汇总统计
        total_compute_ms = model_perf.total_compute_time / 1000.0
        total_memory_ms = model_perf.total_memory_time / 1000.0
        total_transfer_ms = model_perf.total_transfer_time / 1000.0
        total_ms = model_perf.total_time / 1000.0
        
        summary_lines = [
            f"总结统计 / 单层",
            f"  计算时间: {total_compute_ms:>10.3f} ms",
            f"  内存时间: {total_memory_ms:>10.3f} ms",
            f"  传输时间: {total_transfer_ms:>10.3f} ms",
            f"  总耗时:   {total_ms:>10.3f} ms",
        ]
        
        for summary in summary_lines:
            padding = total_width - self._display_width(summary) - 4
            print(f"│ {summary}" + " " * max(padding, 1) + " │")
        
        # 性能瓶颈
        bottleneck = model_perf.get_bottleneck_op()
        if bottleneck:
            layer_name, op_name, op_perf = bottleneck
            print("├" + "─" * (total_width - 2) + "┤")
            bottleneck_text = f"性能瓶颈: {op_name} (总耗时: {op_perf.total_time:.3f} ms)"
            padding = total_width - self._display_width(bottleneck_text) - 4
            print(f"│ {bottleneck_text}" + " " * max(padding, 1) + " │")
        
        # TTFT 与 Throughput
        ttft = model_perf.get_ttft_or_tpot()
        print("├" + "─" * (total_width - 2) + "┤")
        ttft_text = f"TTFT: (耗时: {ttft:.3f} ms)" if model_perf.forward_mode == "EXTEND" else f"TPOT: (耗时: {ttft:.3f} ms)"
        padding = total_width - self._display_width(ttft_text) - 4
        print(f"│ {ttft_text}" + " " * max(padding, 1) + " │")

        throughput = model_perf.get_throughput()
        print("├" + "─" * (total_width - 2) + "┤")
        throughput_txt = f"TPS: (throughput: {throughput:.3f})"
        padding = total_width - self._display_width(throughput_txt) - 4
        print(f"│ {throughput_txt}" + " " * max(padding, 1) + " │")

        print("└" + "─" * (total_width - 2) + "┘")
    
    def save(self, model_perf: ModelPerformance, output_path: str = None) -> None:
        """
        保存性能报告到文件
        
        Args:
            model_perf: 模型性能指标
            output_path: 输出文件路径（可选）
        """
        if output_path:
            # 重定向标准输出到文件
            import sys
            from io import StringIO
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            self.format(model_perf)
            
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"报告已保存到: {output_path}")
        else:
            # 直接打印到控制台
            self.format(model_perf)
