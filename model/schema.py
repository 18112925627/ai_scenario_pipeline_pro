LABELS = ["控糖不推荐","低钠饮食谨慎","儿童不建议","孕妇不建议","心血管风险谨慎","日常适量","素食友好","健身增肌适合"]
def label_to_idx(): return {l:i for i,l in enumerate(LABELS)}
def idx_to_label(): return {i:l for i,l in enumerate(LABELS)}
