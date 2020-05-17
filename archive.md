---
title: archive
permalink: /archive/
layout: page
excerpt: Articles by category
---

<div id="archives">
{% for category in site.categories %}
  <div class="archive-group">
    {% capture category_name %}{{ category | first }}{% endcapture %}
    <div id="#{{ category_name | slugize }}"></div>
    <p></p>

    <h3 class="category-head">{{ category_name }}</h3>
    <a name="{{ category_name | slugize }}"></a>
    {% for post in site.categories[category_name] %}
        <article class="post-item">
        <span class="post-item-date">{{ post.date | date: "%b %d, %Y" }}</span>
        <h5 class="post-item-title">
            <a href="{{ post.url }}">{{ post.title | escape }}</a>
        </h5>
        </article>
    {% endfor %}
  </div>
{% endfor %}
</div>