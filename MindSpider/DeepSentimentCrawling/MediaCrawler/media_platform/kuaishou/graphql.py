# 聲明：本代碼僅供學習和研究目的使用。使用者應遵守以下原則：  
# 1. 不得用於任何商業用途。  
# 2. 使用時應遵守目標平臺的使用條款和robots.txt規則。  
# 3. 不得進行大規模爬取或對平臺造成運營幹擾。  
# 4. 應合理控制請求頻率，避免給目標平臺帶來不必要的負擔。   
# 5. 不得用於任何非法或不當的用途。
#   
# 詳細許可條款請參閱項目根目錄下的LICENSE文件。  
# 使用本代碼即表示您同意遵守上述原則和LICENSE中的所有條款。  


# 快手的數據傳輸是基於GraphQL實現的
# 這個類負責獲取一些GraphQL的schema
from typing import Dict


class KuaiShouGraphQL:
    graphql_queries: Dict[str, str]= {}

    def __init__(self):
        self.graphql_dir = "media_platform/kuaishou/graphql/"
        self.load_graphql_queries()

    def load_graphql_queries(self):
        graphql_files = ["search_query.graphql", "video_detail.graphql", "comment_list.graphql", "vision_profile.graphql","vision_profile_photo_list.graphql","vision_profile_user_list.graphql","vision_sub_comment_list.graphql"]

        for file in graphql_files:
            with open(self.graphql_dir + file, mode="r") as f:
                query_name = file.split(".")[0]
                self.graphql_queries[query_name] = f.read()

    def get(self, query_name: str) -> str:
        return self.graphql_queries.get(query_name, "Query not found")
