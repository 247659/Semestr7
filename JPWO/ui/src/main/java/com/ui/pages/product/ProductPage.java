package com.ui.pages.product;

import org.openqa.selenium.By;
import org.openqa.selenium.NoSuchElementException;
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

    private final By addToCartButton = By.cssSelector("[data-test='add-to-cart']");
    private final By cartBadge = By.cssSelector("[data-test='cart-quantity']");
    private final By cartButton = By.cssSelector("[data-test='nav-cart']");
    private final By productName = By.cssSelector("[data-test='product-name']");
    private final By searchField = By.cssSelector("[data-test='search-query']");

    public ProductPage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    public void open(String productId) {
        driver.get("http://localhost:4200/product/" + productId);
    }

    public void addToCart() {
        WebElement button = wait.until(ExpectedConditions.elementToBeClickable(addToCartButton));
        button.click();
    }

    public void openCartPage() {
        WebElement button = wait.until(ExpectedConditions.elementToBeClickable(cartButton));
        button.click();
    }

    public int getCartCount() {
        try {
            String countText = wait.until(
                    ExpectedConditions.visibilityOfElementLocated(cartBadge)
            ).getText();
            return Integer.parseInt(countText);
        } catch (NoSuchElementException e) {
            return 0;
        }
    }

    public void search(String query) {
        WebElement searchInput = wait.until(ExpectedConditions.visibilityOfElementLocated(searchField));
        searchInput.clear();
        searchInput.sendKeys(query);
        searchInput.submit();
    }

    public List<String> getVisibleProductNames() {
        wait.until(ExpectedConditions.visibilityOfElementLocated(productName));

        List<WebElement> elements = driver.findElements(productName);
        return elements.stream()
                .map(WebElement::getText)
                .filter(name -> !name.isBlank())
                .collect(Collectors.toList());
    }
}
