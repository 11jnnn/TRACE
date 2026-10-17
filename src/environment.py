"""
BookEnvironment: Manages document resources, embeddings, and retrieval.
Supports chapter-level, book-level, and global-level retrieval modes.
"""

import os
import json
import pickle
import torch
from typing import List, Dict, Optional, Tuple
from bisect import bisect_right

from colpali_engine.models import ColPali, ColPaliProcessor
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .config import Config

# TRACE 的“工具箱”：加载页面向量和图、计算相关性、找到图片、返回邻居
class BookEnvironment:
    """
    Environment class that maintains document state, embeddings, graphs, and retrieval models.
    Supports three operational modes: chapter, book, and global.
    """

    def __init__(self, asset_dir: str, image_root: str, config: Config):
        """
        Initialize the BookEnvironment.

        Args:
            asset_dir: Directory containing preprocessed embeddings and graphs
            image_root: Root directory containing document images
            config: Configuration object with model and retrieval settings
        """
        # 资源目录，存放页面向量，图等
        self.asset_dir = asset_dir
        # 原始页面图片的根目录
        self.image_root = image_root
        self.config = config
        #colpali负责吧问题编码成向量
        self.device = config.colpali_device

        print(f"Loading ColPali model on {self.device}...")
        # 加载colpali模型
        self.colpali = ColPali.from_pretrained(
            config.colpali_model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
            local_files_only=True
        ).eval()
        # 加载coloali输入器
        self.colpali_processor = ColPaliProcessor.from_pretrained(
            config.colpali_model_path,
            local_files_only=True
        )

        # Initialize vision client for Navigator
        # 创建页面模型客户端，用来判断页面是否相关、提取证据
        self.vision_client = OpenAIChatCompletionClient(
            api_key=config.navigator.api_key,
            base_url=config.navigator.base_url,
            model=config.navigator.model,
            model_info=config.navigator.model_info or {}
        )

        # Build chapter index: chapter_name -> image_directory
        # 建立章节名到图片目录的映射
        self.chapter_index = self._build_chapter_index()

        # Resource cache
    #当前加载的章节
        self.current_group_id = None
        # 当前范围内的页面向量张量
        self.current_embeddings = None
        #页面建的语义关系# Tensor of embeddings
        self.current_graph = None
        #每个章节在合并页面索引中的位置# Graph adjacency dict
        # 章节指的是第一章，第二章，按照章节，book书籍，global全局存储页面图像
        self.meta_segments = []  
        # 各章节的起始索引# Metadata for index mapping
        self.meta_starts = [] 
        # 已经排序的页面路径# For binary search
        self._dir_cache = {}  # Cache for image file lists

    # 找到每章图片放在哪里
    def _build_chapter_index(self) -> Dict[str, str]:
        """
        Build mapping from chapter name to image directory path.

        Returns:
            Dictionary mapping chapter_name -> full_path_to_images
        """
        chapter_to_path = {}
        if not os.path.exists(self.image_root):
            return {}
        # 找到每本书的文件夹
        for book_name in os.listdir(self.image_root):
            # 得到每本书的路径
            book_path = os.path.join(self.image_root, book_name)
            if os.path.isdir(book_path):
                # 找到这本书的章节文件夹
                for chapter_name in os.listdir(book_path):
                    # 拼接成章节路径
                    chapter_path = os.path.join(book_path, chapter_name)
                    if os.path.isdir(chapter_path):
                        # 得到章节名与章节路径对应列表
                        chapter_to_path[chapter_name] = chapter_path
        return chapter_to_path
# 把指定章节已经生成的页面向量和语义图,从磁盘读入当前环境,供后续检索使用
    def load_document(self, doc_id: str):
        """
        Load a single chapter's resources (chapter mode).

        Args:
            doc_id: Chapter identifier
        """
        #如果当前章节已经加载了,直接结束
        if self.current_group_id == doc_id:
            return
        # 加载该章节
        print(f"Loading chapter: {doc_id}")
        self.current_group_id = doc_id

        #拼出两个资源的路径，一个页面向量，一个语义连接
        emb_path = os.path.join(self.asset_dir, "embeddings", f"{doc_id}.pt")
        graph_path = os.path.join(self.asset_dir, "graphs", f"{doc_id}.pkl")

        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embeddings not found: {emb_path}")

        self.current_embeddings = torch.load(emb_path, map_location="cpu").to(torch.bfloat16)

        #读取页面语义图
        if os.path.exists(graph_path):
            with open(graph_path, 'rb') as f:
                self.current_graph = pickle.load(f)
        else:
            self.current_graph = {}

        # For chapter mode, create simple metadata
        self.meta_segments = [{
            'doc_id': doc_id,
            'start': 0,
            'length': len(self.current_embeddings)
        }]
        self.meta_starts = [0]

    # load_document() 的书籍级／全局级版本：一次加载更大范围内的页面向量、语义图和编号映射。
    def load_resources(self, group_id: str, mode: str = "book"):
        """
        Load resources for book-level or global-level retrieval.

        Args:
            group_id: Book name (for book mode) or "global_graph" (for global mode)
            mode: "book" or "global"
        """
        if self.current_group_id == group_id:
            return

        print(f"Loading resources for {mode} mode: {group_id}")
        self.current_group_id = group_id

        prefix = "book" if mode == "book" else "global"

        # Load embeddings
        emb_path = os.path.join(self.asset_dir, f"{prefix}_embeddings", f"{group_id}.pt")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embeddings not found: {emb_path}")
        self.current_embeddings = torch.load(emb_path, map_location="cpu").to(torch.bfloat16)

        # Load graph
        graph_path = os.path.join(self.asset_dir, f"{prefix}_graphs", f"{group_id}.pkl")
        if os.path.exists(graph_path):
            with open(graph_path, 'rb') as f:
                self.current_graph = pickle.load(f)
        else:
            self.current_graph = {}

        # Load metadata
        meta_path = os.path.join(self.asset_dir, f"{prefix}_embeddings", f"{group_id}_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        with open(meta_path, 'r') as f:
            self.meta_segments = json.load(f)

        self.meta_starts = [seg['start'] for seg in self.meta_segments]
# 把合并后的页面索引，转换成“所属章节＋章节内部的页面索引”。 它只计算编号，不加载图片或向量。
    def _resolve_global_idx(self, global_idx: int) -> Tuple[Optional[str], int]:
        """
        Map global index to (doc_id, local_page_idx).

        Args:
            global_idx: Global page index

        Returns:
            Tuple of (doc_id, local_page_idx) or (None, -1) if invalid
        """
        seg_idx = bisect_right(self.meta_starts, global_idx) - 1
        if seg_idx < 0 or seg_idx >= len(self.meta_segments):
            return None, -1

        segment = self.meta_segments[seg_idx]
        if not (segment['start'] <= global_idx < segment['start'] + segment['length']):
            return None, -1

        local_page_idx = global_idx - segment['start']
        return segment['doc_id'], local_page_idx

    def get_image_path(self, page_idx: int) -> Optional[str]:
        """
        Get image file path for a given page index.

        Args:
            page_idx: Page index (0-based, can be global or local depending on mode)

        Returns:
            Full path to image file, or None if not found
        """
        # For chapter mode
        if len(self.meta_segments) == 1 and self.meta_segments[0]['start'] == 0:
            doc_id = self.meta_segments[0]['doc_id']
            local_idx = page_idx
        else:
            # For book/global mode
            doc_id, local_idx = self._resolve_global_idx(page_idx)
            if not doc_id:
                return None

        chapter_dir = self.chapter_index.get(doc_id)
        if not chapter_dir:
            return None

        # Cache directory listing
        if doc_id not in self._dir_cache:
            files = sorted(
                [f for f in os.listdir(chapter_dir) if f.endswith(('.png', '.jpg'))],
                key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x else 0
            )
            self._dir_cache[doc_id] = [os.path.join(chapter_dir, f) for f in files]

        files = self._dir_cache[doc_id]
        if 0 <= local_idx < len(files):
            return files[local_idx]
        return None

    def index_to_uid(self, page_idx: int) -> str:
        """
        Convert page index to human-readable UID (e.g., "chapter1_page5").

        Args:
            page_idx: Page index (0-based)

        Returns:
            UID string
        """
        if len(self.meta_segments) == 1 and self.meta_segments[0]['start'] == 0:
            doc_id = self.meta_segments[0]['doc_id']
            local_idx = page_idx
        else:
            doc_id, local_idx = self._resolve_global_idx(page_idx)
            if not doc_id:
                return f"unknown_{page_idx}"

        return f"{doc_id}_page{local_idx + 1}"

    # 计算当前问题与已加载的每个页面之间的语义相关性分数。
    def retrieve_page_scores(self, query: str) -> torch.Tensor:
        """
        Calculate similarity scores for all pages given a query.

        Args:
            query: Search query string

        Returns:
            Tensor of scores [num_pages]
        """
        if self.current_embeddings is None:
            return torch.tensor([])

        with torch.no_grad():
            # 把问题转换成模型输入
            batch_query = self.colpali_processor.process_queries([query]).to(self.device)
            query_emb = self.colpali(**batch_query)
            # 得到问题的多向量表示
            doc_emb_device = self.current_embeddings.to(self.device)
            # 计算多向量匹配分数
            scores = self.colpali_processor.score_multi_vector(query_emb, doc_emb_device)

        return scores[0].cpu()

    def retrieve_initial_pages(self, query: str) -> List[int]:
        """
        Retrieve all page indices sorted by similarity (ascending order).

        Args:
            query: Search query string

        Returns:
            List of page indices sorted by score (low to high)
        """
        scores = self.retrieve_page_scores(query)
        if len(scores) == 0:
            return []
        sorted_indices = scores.argsort(descending=False)
        return sorted_indices.tolist()

    def get_semantic_neighbors(self, page_idx: int, k: int = 3) -> List[int]:
        """
        Get semantic neighbors from MoLoRAG graph.

        Args:
            page_idx: Page index
            k: Number of neighbors to return

        Returns:
            List of neighbor page indices
        """
        if not self.current_graph or page_idx not in self.current_graph:
            return []

        neighbors = list(self.current_graph[page_idx])
        return neighbors[:k]
