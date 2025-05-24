{% if obj.display %}
   {% if is_own_page %}
{{ obj.id }}
{{ "=" * obj.id | length }}
   {% endif %}

   {% set visible_children = obj.children | selectattr("display") | list %}
   {% set own_page_children = visible_children | selectattr("type", "in", own_page_types) | list %}

   {# TOCTREE para hijos con su propia página #}
   {% if is_own_page and own_page_children %}
.. toctree::
   :hidden:

      {% for child in own_page_children %}
   {{ child.include_path }}
      {% endfor %}
   {% endif %}

.. py:{{ obj.type }}:: {% if is_own_page %}{{ obj.id }}{% else %}{{ obj.short_name }}{% endif %}{% if obj.args %}({{ obj.args }}){% endif %}

   {% for (args, return_annotation) in obj.overloads %}
      {{ " " * (obj.type | length) }}   {{ obj.short_name }}{% if args %}({{ args }}){% endif %}
   {% endfor %}

   {# Bases e herencia #}
   {% if obj.bases and "show-inheritance" in autoapi_options %}
Bases: {% for base in obj.bases %}{{ base|link_objs }}{% if not loop.last %}, {% endif %}{% endfor %}
   {% endif %}

   {% if "show-inheritance-diagram" in autoapi_options and obj.bases != ["object"] %}
.. autoapi-inheritance-diagram:: {{ obj.obj["full_name"] }}
   :parts: 1
   {% if "private-members" in autoapi_options %}
   :private-bases:
   {% endif %}
   {% endif %}

   {# Docstring principal de la clase u objeto #}
   {% if obj.docstring %}
{{ obj.docstring | indent(3) }}
   {% endif %}

   {# Render inline children que no tienen su propia página #}
   {% for child in visible_children %}
      {% if child.type not in own_page_types %}
{{ child.render() | indent(3) }}
      {% endif %}
   {% endfor %}

   {# --- Secciones resumidas si es la propia página del objeto --- #}
   {% if is_own_page %}
      {% set visible_attributes = visible_children | selectattr("type", "equalto", "attribute") | list %}
      {% if visible_attributes %}
Attributes
----------

.. autoapisummary::

         {% for attr in visible_attributes %}
   {{ attr.id }}
         {% endfor %}
      {% endif %}

      {% set visible_methods = visible_children | selectattr("type", "equalto", "method") | list %}
      {% if visible_methods %}
Methods
-------

.. autoapisummary::

         {% for method in visible_methods %}
   {{ method.id }}
         {% endfor %}
      {% endif %}
   {% endif %}
{% endif %}
