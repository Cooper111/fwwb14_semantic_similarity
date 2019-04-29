# -*- coding: utf-8 -*-
import scrapy
import re
from prob_spider.items import ProbSpiderItem
from urllib import parse
pattern = re.compile(r'"ti">(.*)</a>')


class ProbspiderSpider(scrapy.Spider):
    name = 'probspider'
    # start_urls = ['https://zhidao.baidu.com/search?word=%E8%87%AA%E5%8A%A9%E5%BC%80%E6%88%B7%E5%90%8E%E5%A6%82%E4%BD%95%E8%A1%A5%E5%8A%9E%E4%B8%89%E6%96%B9%E5%AD%98%E7%AE%A1']

    def start_requests(self):
        stdprobs = open('std_prob.txt', 'r').readlines()
        for stdprob in stdprobs:
            url = 'https://zhidao.baidu.com/search?word=' + \
                parse.quote(stdprob.strip())
            yield scrapy.Request(url, callback=lambda response, nowprob=stdprob.strip(): self.parse(response, nowprob))

    def parse(self, response, nowprob):
        for i in response.xpath("//dt[@class='dt mb-4 line']//a").extract():
            item = ProbSpiderItem()
            item['stdprob'] = nowprob
            item['prob'] = pattern.findall(
                i.replace('<em>', '').replace('</em>', ''))[0]
            yield item

        now_page = response.xpath(
            "//div[@class='pager']//b//text()").extract_first()
        if(int(now_page) <= 2):
            next_href = response.xpath(
                "//a[@class='pager-next']//@href").extract_first()
            next_page = response.urljoin(next_href)
            yield scrapy.Request(next_page, callback=lambda response, nowprob=nowprob: self.parse(response, nowprob))
