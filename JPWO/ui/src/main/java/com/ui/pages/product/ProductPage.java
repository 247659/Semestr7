package com.ui.pages.product;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;
import java.util.List;
import java.util.stream.Collectors;

public class ProductPage {

    private final WebDriver driver;
    private final WebDriverWait wait;

    private final By productNameInDashboard = By.cssSelector("[data-test='product-name']");
    private final By productCards = By.cssSelector("a.card");
    private final By searchField = By.cssSelector("[data-test='search-query']");

    public ProductPage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    public void open(String productId) {
        driver.get("http://localhost:4200/product/" + productId);
    }

    public void search(String query) {
        WebElement searchInput = wait.until(ExpectedConditions.visibilityOfElementLocated(searchField));
        searchInput.clear();
        searchInput.sendKeys(query);
        searchInput.submit();
    }

    public List<String> getVisibleProductNames() {
        wait.until(ExpectedConditions.visibilityOfElementLocated(productNameInDashboard));

        List<WebElement> elements = driver.findElements(productNameInDashboard);
        return elements.stream()
                .map(WebElement::getText)
                .filter(name -> !name.isBlank())
                .collect(Collectors.toList());
    }

    public void openFirstProduct() {
        wait.until(ExpectedConditions.visibilityOfElementLocated(productCards));
        List<WebElement> products = driver.findElements(productCards);
        products.getFirst().click();
    }

}
