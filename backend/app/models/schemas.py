from pydantic import BaseModel


class ArticleCreate(BaseModel):
    title: str
    content: str
