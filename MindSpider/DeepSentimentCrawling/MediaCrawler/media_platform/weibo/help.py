# 聲明：本代碼僅供學習和研究目的使用。使用者應遵守以下原則：  
# 1. 不得用於任何商業用途。  
# 2. 使用時應遵守目標平臺的使用條款和robots.txt規則。  
# 3. 不得進行大規模爬取或對平臺造成運營幹擾。  
# 4. 應合理控制請求頻率，避免給目標平臺帶來不必要的負擔。   
# 5. 不得用於任何非法或不當的用途。
#   
# 詳細許可條款請參閱項目根目錄下的LICENSE文件。  
# 使用本代碼即表示您同意遵守上述原則和LICENSE中的所有條款。  


# -*- coding: utf-8 -*-
# @Author  : relakkes@gmail.com
# @Time    : 2023/12/24 17:37
# @Desc    :

from typing import Dict, List


def filter_search_result_card(card_list: List[Dict]) -> List[Dict]:
    """
    過濾微博搜索的結果，只保留card_type爲9類型的數據
    :param card_list:
    :return:
    """
    note_list: List[Dict] = []
    for card_item in card_list:
        if card_item.get("card_type") == 9:
            note_list.append(card_item)
        if len(card_item.get("card_group", [])) > 0:
            card_group = card_item.get("card_group")
            for card_group_item in card_group:
                if card_group_item.get("card_type") == 9:
                    note_list.append(card_group_item)

    return note_list
